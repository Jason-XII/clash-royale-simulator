"""PPO updates overlapped with independent CPU rollout workers.

Each worker collects n_steps using a single policy version and writes into its
slice of temporary shared arrays. There are two sets of arrays: while the GPU
trains on one rollout, the workers fill the other, using the policy from before
that update (one update stale; PPO's clipped ratio corrects for it). SB3 still
computes values/log-probabilities, timeout bootstrapping, GAE and PPO losses.
Step callbacks are delivered after collection (suitable for checkpointing,
not callbacks that change the policy or step the training env mid-rollout).
"""
from collections import deque
from functools import partial
from pathlib import Path
import tempfile
import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import ResultsWriter
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from masked_spatial import LSTM_HIDDEN


FIELDS = ('actions', 'rewards', 'episode_starts', 'values', 'log_probs',
          'advantages', 'returns', 'discounts', 'lambda_discounts', 'hidden')
SEQUENCE_LENGTH = 64  # decisions per training chunk for recurrent policies


def open_shared(descriptors):
    """Map each shared file once. Open mappings survive the path being deleted
    (e.g. by a /tmp cleaner on a cluster), so never reopen by path mid-run."""
    return {key: np.memmap(path, mode='r+', shape=shape, dtype=dtype)
            for key, (path, shape, dtype) in descriptors.items()}


def attach(buffer, arrays, rank=None):
    for key, value in arrays.items():
        if rank is not None:
            value = value[:, rank:rank+1]
        if key.startswith('obs:'):
            buffer.observations[key[4:]] = value
        else:
            setattr(buffer, key, value)


class VariableDiscountBuffer(DictRolloutBuffer):
    """GAE with one discount per transition instead of one per decision.

    Also stores the recurrent policy's LSTM state before each step (`hidden`).
    With `sequence_length` set, minibatches are chunks of that many consecutive
    decisions of one worker, carrying the chunk's starting state.
    """
    sequence_length = 0

    def reset(self):
        super().reset()
        # ponytail: allocated for feed-forward policies too; ~16 MB at 16 x 512 steps.
        self.hidden = np.zeros((self.buffer_size, self.n_envs, 2, LSTM_HIDDEN), dtype=np.float32)
        self.discounts = np.full(
            (self.buffer_size, self.n_envs), self.gamma, dtype=np.float32
        )
        self.lambda_discounts = np.full(
            (self.buffer_size, self.n_envs), self.gae_lambda, dtype=np.float32
        )

    def compute_returns_and_advantage(self, last_values, dones):
        last_values = last_values.clone().cpu().numpy().flatten()
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self.discounts[step] * next_values * next_non_terminal
                - self.values[step]
            )
            last_gae_lam = (
                delta
                + self.discounts[step]
                * self.lambda_discounts[step]
                * next_non_terminal
                * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def get(self, batch_size=None):
        steps = self.sequence_length
        if not steps:
            yield from super().get(batch_size)
            return
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)
            for name in ('actions', 'values', 'log_probs', 'advantages', 'returns',
                         'episode_starts', 'hidden'):
                self.__dict__[name] = self.swap_and_flatten(self.__dict__[name])
            self.generator_ready = True
        # Flat index = worker * buffer_size + step; buffer_size % steps == 0, so
        # no chunk spans two workers.
        firsts = np.random.permutation(np.arange(0, self.buffer_size * self.n_envs, steps))
        per_batch = max(1, (batch_size or len(firsts) * steps) // steps)
        for start in range(0, len(firsts), per_batch):
            first = firsts[start:start + per_batch]
            indices = (first[:, None] + np.arange(steps)).ravel()
            samples = self._get_samples(indices)
            samples.observations['lstm_state'] = self.to_torch(self.hidden[first])
            samples.observations['episode_start'] = self.to_torch(self.episode_starts[indices])
            yield samples


class SharedBuffer(VariableDiscountBuffer):
    def __init__(self, *args, descriptors, rank, **kwargs):
        self.sets, self.rank = [open_shared(d) for d in descriptors], rank
        self.arrays = self.sets[0]          # collect_segment picks the set to fill
        super().__init__(*args, **kwargs)

    def reset(self):
        # Every slot is overwritten before training. Avoid zeroing shared storage.
        self.observations = {}
        attach(self, self.arrays, self.rank)
        self.pos, self.full, self.generator_ready = 0, False, False

    def compute_returns_and_advantage(self, last_values, dones):
        shared_returns = self.returns
        super().compute_returns_and_advantage(last_values, dones)
        # SB3 rebinds returns to a new array; publish it back to shared storage.
        np.copyto(shared_returns, self.returns)
        self.returns = shared_returns


class Events(BaseCallback):
    def _on_step(self):
        position = self.model.rollout_buffer.pos
        steps = np.asarray(
            [info.get('discount_steps', 1) for info in self.locals['infos']],
            dtype=np.float32,
        )
        discounts = np.asarray(
            [info.get('transition_discount', self.model.gamma ** step)
             for info, step in zip(self.locals['infos'], steps)],
            dtype=np.float32,
        )
        expected = self.model.gamma ** steps
        if (not np.isfinite(steps).all()
                or not np.all((steps > 0) | ((steps == 0) & self.locals['dones']))
                or not np.allclose(discounts, expected, rtol=1e-5, atol=1e-7)):
            raise RuntimeError('Environment and PPO transition discounts disagree')
        self.model.rollout_buffer.discounts[position] = discounts
        self.model.rollout_buffer.lambda_discounts[position] = self.model.gae_lambda ** steps
        policy = self.model.policy
        if getattr(policy, 'recurrent', False):
            # Store the state this step's action came from; a finished game restarts it.
            self.model.rollout_buffer.hidden[position] = policy.state_in.cpu().numpy()
            done = torch.as_tensor(self.locals['dones'], dtype=torch.float32).view(-1, 1, 1)
            policy.state_in = policy.next_state * (1 - done)
        self.events.append((self.locals['infos'], self.locals['dones'].copy()))
        return True


class Endpoint(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.local = None
        self.observation = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.observation = obs
        if self.local is not None:
            self.local._last_obs = {key: np.asarray(value)[None] for key, value in obs.items()}
            self.local._last_episode_starts = np.ones(1, dtype=bool)
            self.local.env.episode_returns[:] = 0
            self.local.env.episode_lengths[:] = 0
        return obs, info

    def initialize_collector(self, config, descriptors, rank, started):
        torch.set_num_threads(1)
        # Construction must not perturb the environment or action RNG streams.
        state = torch.get_rng_state()
        env = VecMonitor(DummyVecEnv([lambda: self.env]))
        env.t_start = started
        self.local = PPO(env=env, device='cpu', verbose=0,
                         rollout_buffer_class=SharedBuffer,
                         rollout_buffer_kwargs=dict(descriptors=descriptors, rank=rank), **config)
        torch.set_rng_state(state)
        self.local.set_logger(configure(format_strings=[]))
        self.local._last_obs = {key: np.asarray(value)[None] for key, value in self.observation.items()}
        self.local._last_episode_starts = np.ones(1, dtype=bool)
        self.local.ep_info_buffer, self.local.ep_success_buffer = deque(maxlen=100), deque(maxlen=100)
        self.events = Events()
        self.events.init_callback(self.local)

    def collect_segment(self, weights, version, slot):
        model = self.local
        model.rollout_buffer.arrays = model.rollout_buffer.sets[slot]   # attached by reset()
        model.policy.load_state_dict({key: torch.from_numpy(value) for key, value in weights.items()})
        model.policy.set_training_mode(False)
        self.events.events = []
        started = time.perf_counter()
        model.collect_rollouts(model.env, self.events, model.rollout_buffer, model.n_steps)
        return dict(version=version, events=self.events.events,
                    last_obs=model._last_obs, starts=model._last_episode_starts,
                    seconds=time.perf_counter()-started)


def endpoint(factory):
    return Endpoint(factory())


class ParallelVecEnv(SubprocVecEnv):
    def __init__(self, factories, monitor=None):
        self.storage = None                  # created with the shared files, in prepare
        self.descriptors = None              # two sets: one trained on, one being filled
        self.version = 0
        self.pending = None                  # (version, slot) of the rollout being collected
        self.last_slot = 1                   # the set train() last used; the next fill uses the other
        self.started = time.time()
        self.writer = ResultsWriter(str(monitor), header=dict(t_start=self.started, env_id=None)) if monitor else None
        try:
            super().__init__([partial(endpoint, factory) for factory in factories], start_method='spawn')
        except BaseException:
            self.close()
            raise

    def step_async(self, actions):
        raise RuntimeError('Use ParallelPPO to collect complete rollouts from ParallelVecEnv')

    def prepare(self, model, buffer):
        if self.descriptors is not None:
            return
        arrays = {**{'obs:'+key: value for key, value in buffer.observations.items()},
                  **{key: getattr(buffer, key) for key in FIELDS}}
        self.storage = tempfile.TemporaryDirectory(prefix='ppo-segments-', ignore_cleanup_errors=True)
        self.descriptors = [{}, {}]
        for slot, descriptors in enumerate(self.descriptors):
            for index, (key, value) in enumerate(arrays.items()):
                path = str(Path(self.storage.name)/f'{slot}-{index}.bin')
                np.memmap(path, mode='w+', shape=value.shape, dtype=value.dtype).flush()
                descriptors[key] = (path, value.shape, value.dtype.str)
        self.arrays = [open_shared(descriptors) for descriptors in self.descriptors]
        config = dict(policy=model.policy_class, policy_kwargs=model.policy_kwargs,
                      n_steps=model.n_steps, batch_size=min(model.batch_size, model.n_steps),
                      gamma=model.gamma, gae_lambda=model.gae_lambda, seed=None)
        # Send to every worker first: initialization can proceed concurrently.
        for rank, remote in enumerate(self.remotes):
            remote.send(('env_method', ('initialize_collector', (config, self.descriptors, rank, self.started), {})))
        self.receive()

    def launch(self, policy):
        """Start collecting the next rollout with `policy`'s current weights."""
        weights = {key: value.detach().cpu().numpy() for key, value in policy.state_dict().items()}
        self.version += 1
        slot = 1 - self.last_slot
        for remote in self.remotes:
            remote.send(('env_method', ('collect_segment', (weights, self.version, slot), {})))
        self.pending = (self.version, slot)

    def finish(self):
        """Wait for the rollout in flight; returns (segments, slot)."""
        (version, slot), self.pending = self.pending, None
        segments = self.receive()
        if any(segment['version'] != version for segment in segments):
            raise RuntimeError('Mixed policy versions in rollout')
        self.last_slot = slot
        return segments, slot

    def receive(self):
        deadline = time.monotonic()+1800
        result = []
        for rank, remote in enumerate(self.remotes):
            if not remote.poll(max(0, deadline-time.monotonic())):
                raise TimeoutError(f'Rollout worker {rank} did not respond within 30 minutes')
            try:
                result.append(remote.recv())
            except EOFError as error:
                raise RuntimeError(f'Rollout worker {rank} exited; inspect its traceback') from error
        return result

    def close(self):
        if getattr(self, 'closed', False):
            return
        try:
            for remote in getattr(self, 'remotes', []):
                try:
                    remote.send(('close', None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
            # A rollout still in flight is discarded: don't wait for it.
            deadline = time.monotonic()+(0 if self.pending else 5)
            for process in getattr(self, 'processes', []):
                process.join(timeout=max(0, deadline-time.monotonic()))
            for process in getattr(self, 'processes', []):
                if process.is_alive():
                    process.terminate()
            for process in getattr(self, 'processes', []):
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
        finally:
            self.closed = True
            for remote in getattr(self, 'remotes', []):
                remote.close()
            if self.writer:
                self.writer.close()
            if self.storage is not None:
                self.storage.cleanup()


class ParallelPPO(PPO):
    def train(self):
        if getattr(self.policy, 'recurrent', False):
            steps = min(SEQUENCE_LENGTH, self.n_steps)
            if self.n_steps % steps:
                raise ValueError(f'rollout_steps must be a multiple of {SEQUENCE_LENGTH} for recurrent policies')
            self.rollout_buffer.sequence_length = steps
        super().train()
        for name, value in getattr(self.policy, 'entropy_diagnostics', {}).items():
            self.logger.record(f'exploration/{name}_entropy', float(value.cpu()))

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        if not isinstance(env, ParallelVecEnv):
            raise TypeError('ParallelPPO requires ParallelVecEnv')
        if self.use_sde or n_rollout_steps != self.n_steps:
            raise ValueError('Only the current fixed-length, non-SDE PPO setup is supported')
        self.policy.set_training_mode(False)
        env.prepare(self, rollout_buffer)
        callback.on_rollout_start()
        if env.pending is None:
            env.launch(self.policy)         # first rollout: nothing in flight yet
        segments, slot = env.finish()
        # Workers fill the other set while train() uses this one. They collect with
        # the weights from before this update, so the next rollout is one update stale.
        env.launch(self.policy)
        rollout_buffer.observations = {}
        attach(rollout_buffer, env.arrays[slot])
        expected_returns = rollout_buffer.advantages + rollout_buffer.values
        if not (np.isfinite(expected_returns).all() and
                np.allclose(rollout_buffer.returns, expected_returns, rtol=1e-5, atol=1e-6)):
            raise RuntimeError('Invalid shared value targets; refusing to update the policy')
        rollout_buffer.pos, rollout_buffer.full, rollout_buffer.generator_ready = self.n_steps, True, False
        self._last_obs = {key: np.concatenate([segment['last_obs'][key] for segment in segments])
                          for key in segments[0]['last_obs']}
        self._last_episode_starts = np.concatenate([segment['starts'] for segment in segments])
        # Replay step notifications with the original vector-step counts. Worker
        # policies stay frozen for the whole rollout, as in ordinary PPO.
        for step in range(self.n_steps):
            infos = [segment['events'][step][0][0] for segment in segments]
            dones = np.array([segment['events'][step][1][0] for segment in segments])
            for info in infos:
                if 'episode' in info:
                    if env.writer:
                        env.writer.write_row(info['episode'])
            self.num_timesteps += env.num_envs
            self._update_info_buffer(infos, dones)
            callback.update_locals(locals())
            if not callback.on_step():
                return False
        callback.on_rollout_end()
        return True
