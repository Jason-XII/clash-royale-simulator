"""Synchronous PPO updates with independent CPU rollout workers.

Each worker collects n_steps using a single policy version, writes into its
slice of temporary shared arrays, and waits for the next update. SB3 still
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


FIELDS = ('actions', 'rewards', 'episode_starts', 'values', 'log_probs', 'advantages', 'returns')


def attach(buffer, descriptors, rank=None):
    for key, (path, shape, dtype) in descriptors.items():
        value = np.memmap(path, mode='r+', shape=shape, dtype=dtype)
        if rank is not None:
            value = value[:, rank:rank+1]
        if key.startswith('obs:'):
            buffer.observations[key[4:]] = value
        else:
            setattr(buffer, key, value)


class SharedBuffer(DictRolloutBuffer):
    def __init__(self, *args, descriptors, rank, **kwargs):
        self.descriptors, self.rank = descriptors, rank
        super().__init__(*args, **kwargs)

    def reset(self):
        # Every slot is overwritten before training. Avoid zeroing shared storage.
        self.observations = {}
        attach(self, self.descriptors, self.rank)
        self.pos, self.full, self.generator_ready = 0, False, False

    def compute_returns_and_advantage(self, last_values, dones):
        shared_returns = self.returns
        super().compute_returns_and_advantage(last_values, dones)
        # SB3 rebinds returns to a new array; publish it back to shared storage.
        np.copyto(shared_returns, self.returns)
        self.returns = shared_returns


class Events(BaseCallback):
    def _on_step(self):
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

    def collect_segment(self, weights, version):
        model = self.local
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
        self.storage = tempfile.TemporaryDirectory(prefix='ppo-segments-')
        self.descriptors = None
        self.version = 0
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
        self.descriptors = {}
        for index, (key, value) in enumerate(arrays.items()):
            path = str(Path(self.storage.name)/f'{index}.bin')
            mapped = np.memmap(path, mode='w+', shape=value.shape, dtype=value.dtype)
            self.descriptors[key] = (path, value.shape, value.dtype.str)
            del mapped
        config = dict(policy=model.policy_class, policy_kwargs=model.policy_kwargs,
                      n_steps=model.n_steps, batch_size=min(model.batch_size, model.n_steps),
                      gamma=model.gamma, gae_lambda=model.gae_lambda, seed=None)
        # Send to every worker first: initialization can proceed concurrently.
        for rank, remote in enumerate(self.remotes):
            remote.send(('env_method', ('initialize_collector', (config, self.descriptors, rank, self.started), {})))
        self.receive()

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
            deadline = time.monotonic()+5
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
            self.storage.cleanup()


class ParallelPPO(PPO):
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        if not isinstance(env, ParallelVecEnv):
            raise TypeError('ParallelPPO requires ParallelVecEnv')
        if self.use_sde or n_rollout_steps != self.n_steps:
            raise ValueError('Only the current fixed-length, non-SDE PPO setup is supported')
        self.policy.set_training_mode(False)
        env.prepare(self, rollout_buffer)
        callback.on_rollout_start()
        weights = {key: value.detach().cpu().numpy() for key, value in self.policy.state_dict().items()}
        env.version += 1
        for remote in env.remotes:
            remote.send(('env_method', ('collect_segment', (weights, env.version), {})))
        segments = env.receive()
        if any(segment['version'] != env.version for segment in segments):
            raise RuntimeError('Mixed policy versions in rollout')
        rollout_buffer.observations = {}
        attach(rollout_buffer, env.descriptors)
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
