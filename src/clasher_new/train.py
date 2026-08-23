"""Train a masked PPO policy for the Clash Royale simulator.

The policy deliberately uses a small permutation-invariant set encoder. Entities are
an unordered collection. Placement is hierarchical: the 32x18 arena is split into
6x4 regions of 3x8 cells. The policy learns the card and coarse region, then samples
uniformly from legal cells in that region. The external action remains (slot, tile).
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


class AutoregressiveActionNet(nn.Module):
    """Predict card, region, and shared conditional local-cell logits."""
    n_cards = 5
    n_regions = 24
    n_cells = 24

    def __init__(self, latent_dim):
        super().__init__()
        self.card = nn.Linear(latent_dim, self.n_cards)
        self.region = nn.Linear(latent_dim, self.n_cards * self.n_regions)
        self.context = nn.Sequential(nn.Linear(latent_dim, 128), nn.Tanh())
        self.card_embedding = nn.Embedding(self.n_cards, 16)
        self.region_embedding = nn.Embedding(self.n_regions, 16)
        self.cell = nn.Sequential(
            nn.Linear(128 + 16 + 16, 128), nn.Tanh(), nn.Linear(128, self.n_cells)
        )

    def forward(self, latent):
        batch = latent.shape[0]
        card_logits = self.card(latent)
        region_logits = self.region(latent).view(batch, self.n_cards, self.n_regions)
        context = self.context(latent)[:, None, None, :].expand(
            batch, self.n_cards, self.n_regions, -1
        )
        cards = torch.arange(self.n_cards, device=latent.device)[None, :, None].expand(
            batch, self.n_cards, self.n_regions
        )
        regions = torch.arange(self.n_regions, device=latent.device)[None, None, :].expand(
            batch, self.n_cards, self.n_regions
        )
        cell_input = torch.cat((
            context,
            self.card_embedding(cards),
            self.region_embedding(regions),
        ), dim=-1)
        cell_logits = self.cell(cell_input)
        return card_logits, region_logits, cell_logits


class AutoregressiveActionDistribution(Distribution):
    """Learn card -> region -> exact legal local-cell decisions."""
    n_cards = 5
    n_regions = 24
    n_cells = 24

    def proba_distribution_net(self, latent_dim):
        return AutoregressiveActionNet(latent_dim)

    def proba_distribution(self, params, card_mask=None, placement_mask=None):
        card_logits, region_logits, cell_logits = params
        if card_mask is not None:
            card_logits = card_logits.masked_fill(~card_mask, -torch.inf)
        self.card_distribution = torch.distributions.Categorical(logits=card_logits)

        batch = region_logits.shape[0]
        if placement_mask is None:
            cell_mask = torch.ones(
                (batch, 4, self.n_regions, self.n_cells),
                dtype=torch.bool, device=region_logits.device
            )
        else:
            mask = placement_mask.view(batch, 4, 4, 8, 6, 3)
            mask = mask.permute(0, 1, 2, 4, 3, 5)
            cell_mask = mask.reshape(batch, 4, self.n_regions, self.n_cells)

        region_mask = cell_mask.any(dim=-1)
        # Give no-op a single fixed conditional outcome, whose log probability
        # and entropy are zero. Active cards retain their legal masks.
        no_op_region = torch.zeros(
            (batch, 1, self.n_regions), dtype=torch.bool, device=region_logits.device
        )
        no_op_region[..., 0] = True
        no_op_cell = torch.zeros(
            (batch, 1, self.n_regions, self.n_cells),
            dtype=torch.bool, device=region_logits.device
        )
        no_op_cell[..., 0, 0] = True
        self.region_mask = torch.cat((no_op_region, region_mask), dim=1)
        self.cell_mask = torch.cat((no_op_cell, cell_mask), dim=1)

        region_logits = region_logits.masked_fill(~self.region_mask, -torch.inf)
        cell_logits = cell_logits.masked_fill(~self.cell_mask, -torch.inf)
        # Fully unavailable card rows have zero card probability, but must stay
        # finite so Categorical entropy remains well-defined.
        region_logits = torch.where(
            self.region_mask.any(dim=-1, keepdim=True), region_logits,
            torch.zeros_like(region_logits)
        )
        cell_logits = torch.where(
            self.cell_mask.any(dim=-1, keepdim=True), cell_logits,
            torch.zeros_like(cell_logits)
        )
        self.region_logits, self.cell_logits = region_logits, cell_logits
        return self

    @staticmethod
    def _tile_to_indices(tiles):
        y, x = tiles // 18, tiles % 18
        return (y // 8) * 6 + x // 3, (y % 8) * 3 + x % 3

    @staticmethod
    def _cell_to_tile(regions, cells):
        region_y, region_x = regions // 6, regions % 6
        local_y, local_x = cells // 3, cells % 3
        return (region_y * 8 + local_y) * 18 + region_x * 3 + local_x

    @staticmethod
    def _rows(cards):
        return torch.arange(cards.shape[0], device=cards.device)

    def _region_distribution(self, cards):
        return torch.distributions.Categorical(
            logits=self.region_logits[self._rows(cards), cards]
        )

    def _cell_distribution(self, cards, regions):
        return torch.distributions.Categorical(
            logits=self.cell_logits[self._rows(cards), cards, regions]
        )

    def log_prob(self, actions):
        cards, tiles = actions[:, 0].long(), actions[:, 1].long()
        active = (cards != 0).float()
        regions, cells = self._tile_to_indices(tiles)
        return (
            self.card_distribution.log_prob(cards)
            + active * self._region_distribution(cards).log_prob(regions)
            + active * self._cell_distribution(cards, regions).log_prob(cells)
        )

    def entropy(self):
        card_probs = self.card_distribution.probs
        region_dist = torch.distributions.Categorical(logits=self.region_logits)
        cell_dist = torch.distributions.Categorical(logits=self.cell_logits)
        conditional = region_dist.entropy() + (
            region_dist.probs * cell_dist.entropy()
        ).sum(dim=-1)
        return self.card_distribution.entropy() + (
            card_probs * conditional
        ).sum(dim=-1)

    def sample(self):
        cards = self.card_distribution.sample()
        regions = self._region_distribution(cards).sample()
        cells = self._cell_distribution(cards, regions).sample()
        tiles = self._cell_to_tile(regions, cells)
        return torch.stack((cards, tiles * (cards != 0)), dim=1)

    def mode(self):
        cards = self.card_distribution.probs.argmax(dim=1)
        regions = self._region_distribution(cards).probs.argmax(dim=1)
        cells = self._cell_distribution(cards, regions).probs.argmax(dim=1)
        tiles = self._cell_to_tile(regions, cells)
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
        self.action_dist = AutoregressiveActionDistribution()
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


class ActionDiagnosticsCallback(BaseCallback):
    """Log action validity and reward components needed to debug learning."""
    def __init__(self, log_every=4096, verbose=0):
        super().__init__(verbose)
        self.log_every = log_every
        self.count = self.no_ops = self.invalid = 0
        self.red_damage = self.blue_damage = 0.0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            self.count += 1
            self.no_ops += int(info.get("no_op", False))
            self.invalid += int(not info.get("no_op", False) and not info.get("deployment_succeeded", False))
            self.red_damage += info.get("red_hp_damage", 0.0)
            self.blue_damage += info.get("blue_hp_damage", 0.0)
        if self.count >= self.log_every:
            self.logger.record("debug/no_op_rate", self.no_ops / self.count)
            self.logger.record("debug/invalid_deployment_rate", self.invalid / self.count)
            self.logger.record("debug/red_damage_per_decision", self.red_damage / self.count)
            self.logger.record("debug/blue_damage_per_decision", self.blue_damage / self.count)
            self.count = self.no_ops = self.invalid = 0
            self.red_damage = self.blue_damage = 0.0
        return True


class EvalCallback(BaseCallback):
    def __init__(self, eval_freq=20_000, episodes=50, verbose=0):
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
    checkpoint = os.environ.get("CHECKPOINT", "autoregressive_checkpoint")
    kwargs = {"features_extractor_class": CRSetExtractor, "net_arch": {"pi": [], "vf": []}}
    model = PPO(
        CRPolicy, vec_env, policy_kwargs=kwargs, n_steps=256, batch_size=256,
        n_epochs=4, learning_rate=3e-4, ent_coef=0.002, vf_coef=0.5,
        clip_range=0.2, gamma=0.995, gae_lambda=0.95, max_grad_norm=0.5,
        verbose=1, tensorboard_log="./cr_logs/tensorboard/",
    )
    try:
        model.learn(total_timesteps=int(os.environ.get("TOTAL_STEPS", "1000000")),
                    callback=[CheckpointCallback(save_freq=max(10000 // n_envs, 1), save_path="./cr_logs/", name_prefix="autoregressive"), ActionDiagnosticsCallback(), EvalCallback()])
    finally:
        model.save(checkpoint)
        vec_env.close()
