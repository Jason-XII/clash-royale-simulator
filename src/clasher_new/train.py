from environment import CREnv, random_strategy, entity_names

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn
import torch.nn.functional as F
import torch

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


class AutoregressiveDistribution(Distribution):
    def __init__(self):
        super().__init__()
        self.card_distribution = None
        self.tile_logits = None

    def proba_distribution_net(self, latent_dim):
        return nn.Linear(latent_dim, 5 + 5 * 32 * 18)

    def proba_distribution(self, action_logits, card_mask=None):
        if card_mask is not None:
            action_logits = action_logits.clone()
            action_logits[:, :5] = action_logits[:, :5].masked_fill(
                ~card_mask, -torch.inf
            )
        self.card_distribution = torch.distributions.Categorical(
            logits=action_logits[:, :5]
        )
        self.tile_logits = action_logits[:, 5:].view(-1, 5, 32 * 18)
        return self

    def tile_distribution(self, card_slots):
        batch_indices = torch.arange(card_slots.shape[0], device=card_slots.device)
        return torch.distributions.Categorical(
            logits=self.tile_logits[batch_indices, card_slots]
        )

    def log_prob(self, actions):
        card_slots = actions[:, 0].long()
        tiles = actions[:, 1].long()
        log_prob = self.card_distribution.log_prob(card_slots)
        tile_log_prob = self.tile_distribution(card_slots).log_prob(tiles)
        return log_prob + (card_slots != 0) * tile_log_prob

    def entropy(self):
        card_entropy = self.card_distribution.entropy()
        tile_entropy = torch.distributions.Categorical(
            logits=self.tile_logits
        ).entropy()
        return card_entropy + (
            self.card_distribution.probs[:, 1:] * tile_entropy[:, 1:]
        ).sum(dim=1)

    def sample(self):
        card_slots = self.card_distribution.sample()
        tiles = self.tile_distribution(card_slots).sample()
        tiles = tiles * (card_slots != 0)
        return torch.stack([card_slots, tiles], dim=1)

    def mode(self):
        card_slots = torch.argmax(self.card_distribution.probs, dim=1)
        tiles = torch.argmax(self.tile_distribution(card_slots).probs, dim=1)
        tiles = tiles * (card_slots != 0)
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

    def _get_action_dist_from_latent(self, latent_pi, card_mask=None):
        return self.action_dist.proba_distribution(
            self.action_net(latent_pi), card_mask
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
        distribution = self._get_action_dist_from_latent(latent_pi, card_mask)
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
        distribution = self._get_action_dist_from_latent(latent_pi, card_mask)
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
        return self._get_action_dist_from_latent(latent_pi, card_mask)


class WeightsCopyingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        if self.num_timesteps % 50000 == 0:
            opponent.policy.load_state_dict(self.model.policy.state_dict())
        return True

class RandomEvalCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.num_timesteps % 50000 == 0:
            rewards = []
            eval_env = CREnv(opponent_model=lambda obs: random_strategy(obs))
            for i in range(5):
                obs, _ = eval_env.reset()
                done = False
                total_reward = 0
                while not done:
                    action, _ = self.model.predict(obs)
                    obs, reward, termination, truncation, info = eval_env.step(action)
                    done = termination or truncation
                    total_reward += reward

                rewards.append(total_reward)
            self.logger.record("eval/mean_reward_vs_random", sum(rewards)/len(rewards))
        return True


if __name__ == '__main__':
    env = CREnv(opponent_model=random_strategy)
    policy_kwargs = {
        "features_extractor_class": CRTransformerExtractor,
        "net_arch": {"pi": [], "vf": []},
    }
    model = PPO(CRAutoregressivePolicy, env, policy_kwargs=policy_kwargs)
    cb = CheckpointCallback(save_freq=10_000, save_path="./cr_logs/", name_prefix="cr")
    try:
        model.learn(total_timesteps=1_000_000, reset_num_timesteps=False, callback=[cb])
    finally:
        print('Saving model.')
        model.save('cr_discrete')
