from environment import (
    CREnv,
    bridge_random_strategy,
    defensive_random_strategy,
    entity_names,
    legal_random_strategy,
    mixed_random_strategy,
    patient_random_strategy,
)

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import random
from pathlib import Path

import time


class CRTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        self.entity_embedding = nn.Embedding(len(entity_names), features_dim)
        self.card_type_embedding = nn.Embedding(4, features_dim)
        self.owner_embedding = nn.Embedding(2, features_dim)
        self.hand_embedding = nn.Embedding(len(entity_names), features_dim)
        self.entity_features = nn.Linear(14, features_dim)
        self.state_features = nn.Linear(4, features_dim)
        self.game_token = nn.Parameter(torch.zeros(1, 1, features_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=features_dim,
            nhead=4,
            dim_feedforward=features_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)

    def forward(self, observation):
        entity_ids = observation["entity_ids"].long()
        entity_features = observation["entity_features"].float()
        entity_mask = observation["entity_mask"].bool()
        hand = observation["hand"].long()
        state = observation["state"].float()

        entity_tokens = (
            self.entity_embedding(entity_ids[..., 0])
            + self.card_type_embedding(entity_ids[..., 1])
            + self.owner_embedding(entity_ids[..., 2])
            + self.entity_features(entity_features)
        )
        hand_tokens = self.hand_embedding(hand)
        state_token = self.state_features(state).unsqueeze(1)
        game_token = self.game_token.expand(entity_ids.shape[0], -1, -1)
        tokens = torch.cat(
            [game_token, state_token, hand_tokens, entity_tokens], dim=1
        )
        padding_mask = torch.cat(
            [
                torch.zeros(
                    (entity_ids.shape[0], 7),
                    dtype=torch.bool,
                    device=entity_ids.device,
                ),
                ~entity_mask,
            ],
            dim=1,
        )
        return self.transformer(tokens, src_key_padding_mask=padding_mask)[:, 0]


class FactorizedActionNet(nn.Module):
    def __init__(self, latent_dim, embedding_dim=64):
        super().__init__()
        self.card_logits = nn.Linear(latent_dim, 5)
        self.context = nn.Linear(latent_dim, embedding_dim)
        self.card_embedding = nn.Parameter(torch.randn(5, embedding_dim))
        self.y_embedding = nn.Parameter(torch.randn(32, embedding_dim))
        self.x_embedding = nn.Parameter(torch.randn(18, embedding_dim))

    def forward(self, latent):
        context = self.context(latent)
        card_context = context.unsqueeze(1) + self.card_embedding.unsqueeze(0)
        y_logits = torch.einsum("bcd,yd->bcy", card_context, self.y_embedding)
        xy_context = card_context.unsqueeze(2) + self.y_embedding.view(1, 1, 32, -1)
        x_logits = torch.einsum("bcyd,xd->bcyx", xy_context, self.x_embedding)
        return self.card_logits(latent), y_logits, x_logits


class AutoregressiveDistribution(Distribution):
    def __init__(self):
        super().__init__()
        self.card_distribution = None
        self.y_logits = None
        self.x_logits = None

    def proba_distribution_net(self, latent_dim):
        return FactorizedActionNet(latent_dim)

    def proba_distribution(self, action_logits, card_mask=None, placement_mask=None):
        card_logits, y_logits, x_logits = action_logits
        if card_mask is not None:
            card_logits = card_logits.masked_fill(~card_mask, -torch.inf)
        if placement_mask is not None:
            no_op_mask = torch.ones(
                (placement_mask.shape[0], 1, 32 * 18),
                dtype=torch.bool,
                device=placement_mask.device,
            )
            tile_mask = torch.cat([no_op_mask, placement_mask], dim=1)
            tile_mask = tile_mask.view(-1, 5, 32, 18)
            valid_y = tile_mask.any(dim=-1)
            y_logits = y_logits.masked_fill(~valid_y, -torch.inf)
            x_logits = x_logits.masked_fill(~tile_mask, -torch.inf)
            y_logits = torch.where(valid_y, y_logits, torch.zeros_like(y_logits))
            x_logits = torch.where(
                valid_y.unsqueeze(-1), x_logits, torch.zeros_like(x_logits)
            )
        self.card_distribution = torch.distributions.Categorical(logits=card_logits)
        self.y_logits = y_logits
        self.x_logits = x_logits
        return self

    def y_distribution(self, card_slots):
        batch_indices = torch.arange(card_slots.shape[0], device=card_slots.device)
        return torch.distributions.Categorical(
            logits=self.y_logits[batch_indices, card_slots]
        )

    def x_distribution(self, card_slots, ys):
        batch_indices = torch.arange(card_slots.shape[0], device=card_slots.device)
        return torch.distributions.Categorical(
            logits=self.x_logits[batch_indices, card_slots, ys]
        )

    def log_prob(self, actions):
        card_slots = actions[:, 0].long()
        tiles = actions[:, 1].long()
        ys = tiles // 18
        xs = tiles % 18
        active = card_slots != 0
        log_prob = self.card_distribution.log_prob(card_slots)
        log_prob = log_prob + active * self.y_distribution(card_slots).log_prob(ys)
        log_prob = log_prob + active * self.x_distribution(card_slots, ys).log_prob(xs)
        return log_prob

    def entropy(self):
        card_probs = self.card_distribution.probs
        card_entropy = self.card_distribution.entropy()
        y_entropy = torch.distributions.Categorical(logits=self.y_logits).entropy()
        x_entropy = torch.distributions.Categorical(logits=self.x_logits).entropy()
        conditional_entropy = (card_probs[:, 1:] * y_entropy[:, 1:]).sum(dim=1)
        y_probs = torch.softmax(self.y_logits[:, 1:], dim=-1)
        conditional_entropy += (
            card_probs[:, 1:, None] * y_probs * x_entropy[:, 1:]
        ).sum(dim=(1, 2))
        return card_entropy + conditional_entropy

    def sample(self):
        card_slots = self.card_distribution.sample()
        ys = self.y_distribution(card_slots).sample()
        xs = self.x_distribution(card_slots, ys).sample()
        tiles = ys * 18 + xs
        tiles = tiles * (card_slots != 0)
        return torch.stack([card_slots, tiles], dim=1)

    def mode(self):
        card_slots = torch.argmax(self.card_distribution.probs, dim=1)
        ys = torch.argmax(self.y_distribution(card_slots).probs, dim=1)
        xs = torch.argmax(self.x_distribution(card_slots, ys).probs, dim=1)
        tiles = (ys * 18 + xs) * (card_slots != 0)
        return torch.stack([card_slots, tiles], dim=1)

    def actions_from_params(self, action_logits, deterministic=False):
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits):
        actions = self.actions_from_params(action_logits)
        return actions, self.log_prob(actions)


class CRAutoregressivePolicy(ActorCriticPolicy):
    def _build(self, lr_schedule):
        self._build_mlp_extractor()
        self.action_dist = AutoregressiveDistribution()
        self.action_net = self.action_dist.proba_distribution_net(
            self.mlp_extractor.latent_dim_pi
        )
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_action_dist_from_latent(
            self, latent_pi, card_mask=None, placement_mask=None):
        return self.action_dist.proba_distribution(
            self.action_net(latent_pi), card_mask, placement_mask
        )

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        card_mask = torch.cat(
            [
                torch.ones_like(obs["hand_mask"][:, :1]),
                obs["hand_mask"][:, :4],
            ],
            dim=1,
        ).bool()
        distribution = self._get_action_dist_from_latent(
            latent_pi, card_mask, obs["placement_mask"].bool()
        )
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, values, distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        card_mask = torch.cat(
            [
                torch.ones_like(obs["hand_mask"][:, :1]),
                obs["hand_mask"][:, :4],
            ],
            dim=1,
        ).bool()
        distribution = self._get_action_dist_from_latent(
            latent_pi, card_mask, obs["placement_mask"].bool()
        )
        return (
            self.value_net(latent_vf),
            distribution.log_prob(actions),
            distribution.entropy(),
        )

    def get_distribution(self, obs):
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        card_mask = torch.cat(
            [
                torch.ones_like(obs["hand_mask"][:, :1]),
                obs["hand_mask"][:, :4],
            ],
            dim=1,
        ).bool()
        return self._get_action_dist_from_latent(
            latent_pi, card_mask, obs["placement_mask"].bool()
        )


class WeightsCopyingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        if self.num_timesteps % 50000 == 0:
            opponent.policy.load_state_dict(self.model.policy.state_dict())
        return True

class RandomEvalCallback(BaseCallback):
    def __init__(self, eval_freq=20_000, n_eval_episodes=50, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.last_eval = 0
        self.best_win_rate = -1.0
        self.strategies = [
            legal_random_strategy,
            patient_random_strategy,
            bridge_random_strategy,
            defensive_random_strategy,
            mixed_random_strategy,
        ]
        Path("cr_logs/best").mkdir(parents=True, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps - self.last_eval < self.eval_freq:
            return True
        self.last_eval = self.num_timesteps
        wins = 0
        rewards = []
        no_ops = 0
        invalid = 0
        decisions = 0

        for episode in range(self.n_eval_episodes):
            eval_env = CREnv(self.strategies[episode % len(self.strategies)])
            obs, _ = eval_env.reset(seed=episode)
            total_reward = 0.0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                total_reward += reward
                decisions += 1
                no_ops += int(info["no_op"])
                invalid += int(not info["no_op"] and not info["deployment_succeeded"])
                done = terminated or truncated
            wins += int(eval_env.battle.winner == 0)
            rewards.append(total_reward)
            eval_env.close()

        win_rate = wins / self.n_eval_episodes
        self.logger.record("eval/win_rate", win_rate)
        self.logger.record("eval/mean_reward", sum(rewards) / len(rewards))
        self.logger.record("eval/no_op_rate", no_ops / decisions)
        self.logger.record("eval/invalid_deployment_rate", invalid / decisions)
        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            self.model.save("cr_logs/best/best_model")
        return True


def make_env(rank):
    strategies = [
        legal_random_strategy,
        patient_random_strategy,
        bridge_random_strategy,
        defensive_random_strategy,
    ]

    def factory():
        random.seed(10_000 + rank)
        torch.manual_seed(10_000 + rank)
        return CREnv(strategies[rank % len(strategies)])

    return factory


if __name__ == "__main__":
    n_envs = int(os.environ.get("N_ENVS", min(8, os.cpu_count() or 1)))
    env = SubprocVecEnv(
        [make_env(rank) for rank in range(n_envs)],
        start_method="spawn",
    )
    env = VecMonitor(env)

    checkpoint = "factorized_masked_checkpoint"
    policy_kwargs = {
        "features_extractor_class": CRTransformerExtractor,
        "net_arch": {"pi": [], "vf": []},
    }

    if os.path.exists(checkpoint + ".zip"):
        model = PPO.load(checkpoint, env=env)
    else:
        model = PPO(
            CRAutoregressivePolicy,
            env,
            policy_kwargs=policy_kwargs,
            n_steps=512,
            batch_size=256,
            ent_coef=0.005,
            clip_range=0.1,
            learning_rate=1e-4,
            verbose=1,
            tensorboard_log="./cr_logs/tensorboard/",
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // n_envs, 1),
        save_path="./cr_logs/",
        name_prefix="factorized_masked",
    )
    eval_callback = RandomEvalCallback(eval_freq=20_000, n_eval_episodes=50)

    try:
        model.learn(
            total_timesteps=1_000_000,
            reset_num_timesteps=False,
            callback=[checkpoint_callback, eval_callback],
        )
    finally:
        model.save(checkpoint)
        env.close()
