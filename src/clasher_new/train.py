"""Train a masked PPO policy for the Clash Royale simulator.

The policy deliberately uses a small permutation-invariant set encoder. Entities are
an unordered collection, and a 32x18 placement is one decision, so treating the
arena as a long transformer sequence and factorizing y/x adds optimization noise
without adding useful information.
"""
from environment import (
    CREnv, bridge_random_strategy, defensive_random_strategy,
    legal_random_strategy, mixed_random_strategy, patient_random_strategy,
    entity_names,
)

import os
from pathlib import Path
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


class CRSetExtractor(BaseFeaturesExtractor):
    """Encode entities as a masked set, avoiding dependence on entity ordering."""
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.entity_embedding = nn.Embedding(len(entity_names), 32)
        self.type_embedding = nn.Embedding(4, 8)
        self.owner_embedding = nn.Embedding(2, 8)
        self.entity_mlp = nn.Sequential(
            nn.Linear(14 + 32 + 8 + 8, 96), nn.LayerNorm(96), nn.Tanh(),
            nn.Linear(96, 96), nn.Tanh(),
        )
        self.hand_embedding = nn.Embedding(len(entity_names), 32)
        self.head = nn.Sequential(
            nn.Linear(96 * 2 + 32 + 4, features_dim), nn.LayerNorm(features_dim), nn.Tanh()
        )

    def forward(self, obs):
        ids = obs["entity_ids"].long()
        features = obs["entity_features"].float()
        mask = obs["entity_mask"].float().unsqueeze(-1)
        tokens = self.entity_mlp(torch.cat((
            features, self.entity_embedding(ids[..., 0]),
            self.type_embedding(ids[..., 1]), self.owner_embedding(ids[..., 2]),
        ), dim=-1))
        denom = mask.sum(dim=1).clamp_min(1.0)
        mean = (tokens * mask).sum(dim=1) / denom
        masked = tokens.masked_fill(mask == 0, -1e9).amax(dim=1)
        hand = self.hand_embedding(obs["hand"].long()).mean(dim=1)
        return self.head(torch.cat((mean, masked, hand, obs["state"].float()), dim=-1))


class JointActionNet(nn.Module):
    def __init__(self, latent_dim, n_cards=5, n_tiles=32 * 18):
        super().__init__()
        self.card = nn.Linear(latent_dim, n_cards)
        self.tiles = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.Tanh(), nn.Linear(256, n_cards * n_tiles)
        )
        self.n_cards, self.n_tiles = n_cards, n_tiles

    def forward(self, latent):
        return self.card(latent), self.tiles(latent).view(-1, self.n_cards, self.n_tiles)


class JointActionDistribution(Distribution):
    def proba_distribution_net(self, latent_dim):
        return JointActionNet(latent_dim)

    def proba_distribution(self, params, card_mask=None, placement_mask=None):
        card_logits, tile_logits = params
        if card_mask is not None:
            card_logits = card_logits.masked_fill(~card_mask, -torch.inf)
        if placement_mask is not None:
            no_op = torch.ones((placement_mask.shape[0], 1, placement_mask.shape[2]),
                               dtype=torch.bool, device=placement_mask.device)
            tile_mask = torch.cat((no_op, placement_mask), dim=1)
            tile_logits = tile_logits.masked_fill(~tile_mask, -torch.inf)
            # Categorical cannot normalize an entirely masked row. Those rows
            # belong to cards with zero card probability, so their values are
            # irrelevant; make them finite for stable entropy computation.
            has_tile = tile_mask.any(dim=-1, keepdim=True)
            tile_logits = torch.where(has_tile, tile_logits, torch.zeros_like(tile_logits))
        self.card_distribution = torch.distributions.Categorical(logits=card_logits)
        self.tile_logits = tile_logits
        return self

    def _tile_distribution(self, cards):
        rows = torch.arange(cards.shape[0], device=cards.device)
        return torch.distributions.Categorical(logits=self.tile_logits[rows, cards])

    def log_prob(self, actions):
        cards, tiles = actions[:, 0].long(), actions[:, 1].long()
        active = (cards != 0).float()
        return self.card_distribution.log_prob(cards) + active * self._tile_distribution(cards).log_prob(tiles)

    def entropy(self):
        card_probs = self.card_distribution.probs
        tile_entropy = torch.distributions.Categorical(logits=self.tile_logits).entropy()
        return self.card_distribution.entropy() + (card_probs * tile_entropy).sum(dim=1)

    def sample(self):
        cards = self.card_distribution.sample()
        tiles = self._tile_distribution(cards).sample()
        return torch.stack((cards, tiles * (cards != 0)), dim=1)

    def mode(self):
        cards = self.card_distribution.probs.argmax(dim=1)
        tiles = self._tile_distribution(cards).probs.argmax(dim=1)
        return torch.stack((cards, tiles * (cards != 0)), dim=1)

    def actions_from_params(self, params, deterministic=False):
        self.proba_distribution(params)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, params):
        actions = self.actions_from_params(params)
        return actions, self.log_prob(actions)


class CRPolicy(ActorCriticPolicy):
    def _build(self, lr_schedule):
        self._build_mlp_extractor()
        self.action_dist = JointActionDistribution()
        self.action_net = self.action_dist.proba_distribution_net(self.mlp_extractor.latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _distribution(self, obs):
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        card_mask = torch.cat((torch.ones_like(obs["hand_mask"][:, :1]), obs["hand_mask"][:, :4]), dim=1).bool()
        return self.action_dist.proba_distribution(
            self.action_net(latent_pi), card_mask, obs["placement_mask"].bool())

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        dist = self._distribution(obs)
        actions = dist.get_actions(deterministic)
        return actions, self.value_net(latent_vf), dist.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        dist = self._distribution(obs)
        return self.value_net(latent_vf), dist.log_prob(actions), dist.entropy()

    def get_distribution(self, obs):
        return self._distribution(obs)


class EvalCallback(BaseCallback):
    def __init__(self, eval_freq=20_000, episodes=10, verbose=0):
        super().__init__(verbose)
        self.eval_freq, self.episodes, self.last = eval_freq, episodes, 0
        self.strategies = [legal_random_strategy, patient_random_strategy,
                           bridge_random_strategy, defensive_random_strategy, mixed_random_strategy]
        self.best = -1.0
        Path("cr_logs/best").mkdir(parents=True, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps - self.last < self.eval_freq:
            return True
        self.last = self.num_timesteps
        wins = 0
        for i in range(self.episodes):
            env = CREnv(self.strategies[i % len(self.strategies)], speed=1.0)
            obs, _ = env.reset(seed=10000 + i)
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(action)
                done = term or trunc
            wins += int(env.battle.winner == 0)
        rate = wins / self.episodes
        self.logger.record("eval/win_rate", rate)
        if rate > self.best:
            self.best = rate
            self.model.save("cr_logs/best/best_model")
        return True


def make_env(rank):
    # A stationary opponent makes the early learning signal interpretable.
    def factory():
        torch.manual_seed(1000 + rank)
        return CREnv(legal_random_strategy, speed=1.0)
    return factory


if __name__ == "__main__":
    n_envs = int(os.environ.get("N_ENVS", min(4, os.cpu_count() or 1)))
    vec_env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method="spawn"))
    checkpoint = os.environ.get("CHECKPOINT", "joint_masked_checkpoint")
    kwargs = {"features_extractor_class": CRSetExtractor, "net_arch": {"pi": [], "vf": []}}
    model = PPO(
        CRPolicy, vec_env, policy_kwargs=kwargs, n_steps=256, batch_size=256,
        n_epochs=4, learning_rate=3e-4, ent_coef=0.002, vf_coef=0.5,
        clip_range=0.2, gamma=0.995, gae_lambda=0.95, max_grad_norm=0.5,
        verbose=1, tensorboard_log="./cr_logs/tensorboard/",
    )
    try:
        model.learn(total_timesteps=int(os.environ.get("TOTAL_STEPS", "1000000")),
                    callback=[CheckpointCallback(save_freq=max(10000 // n_envs, 1), save_path="./cr_logs/", name_prefix="joint_masked"), EvalCallback()])
    finally:
        model.save(checkpoint)
        vec_env.close()
