from environment import CREnv, random_strategy

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
import torch
import torch.nn as nn

from train import CRFeatureExtractor


class AutoregressiveActionNet(nn.Module):
    n_cards = 5
    n_regions = 24
    n_cells = 24

    def __init__(self, latent_dim):
        super().__init__()
        self.card_head = nn.Linear(latent_dim, self.n_cards)
        self.region_head = nn.Linear(latent_dim, self.n_cards * self.n_regions)
        self.context = nn.Sequential(nn.Linear(latent_dim, 128), nn.Tanh())
        self.card_embedding = nn.Embedding(self.n_cards, 16)
        self.region_embedding = nn.Embedding(self.n_regions, 16)
        self.cell_head = nn.Sequential(
            nn.Linear(128 + 16 + 16, 128),
            nn.Tanh(),
            nn.Linear(128, self.n_cells),
        )

    def forward(self, latent):
        batch_size = latent.shape[0]
        card_logits = self.card_head(latent)
        region_logits = self.region_head(latent).view(
            batch_size, self.n_cards, self.n_regions
        )

        context = self.context(latent)[:, None, None, :].expand(
            batch_size, self.n_cards, self.n_regions, -1
        )
        cards = torch.arange(self.n_cards, device=latent.device)[None, :, None]
        cards = cards.expand(batch_size, self.n_cards, self.n_regions)
        regions = torch.arange(self.n_regions, device=latent.device)[None, None, :]
        regions = regions.expand(batch_size, self.n_cards, self.n_regions)
        cell_input = torch.cat(
            (
                context,
                self.card_embedding(cards),
                self.region_embedding(regions),
            ),
            dim=-1,
        )
        cell_logits = self.cell_head(cell_input)
        return card_logits, region_logits, cell_logits


class AutoregressiveActionDistribution(Distribution):
    n_cards = 5
    n_regions = 24
    n_cells = 24

    def proba_distribution_net(self, latent_dim):
        return AutoregressiveActionNet(latent_dim)

    def proba_distribution(self, parameters):
        card_logits, self.region_logits, self.cell_logits = parameters
        self.card_distribution = torch.distributions.Categorical(logits=card_logits)
        return self

    @staticmethod
    def _rows(values):
        return torch.arange(values.shape[0], device=values.device)

    def _region_distribution(self, cards):
        return torch.distributions.Categorical(
            logits=self.region_logits[self._rows(cards), cards]
        )

    def _cell_distribution(self, cards, regions):
        return torch.distributions.Categorical(
            logits=self.cell_logits[self._rows(cards), cards, regions]
        )

    @staticmethod
    def _position_to_region_cell(y, x):
        region = (y // 8) * 6 + x // 3
        cell = (y % 8) * 3 + x % 3
        return region, cell

    @staticmethod
    def _region_cell_to_position(region, cell):
        region_y, region_x = region // 6, region % 6
        local_y, local_x = cell // 3, cell % 3
        return region_y * 8 + local_y, region_x * 3 + local_x

    def log_prob(self, actions):
        cards = actions[:, 0].long()
        y = actions[:, 1].long()
        x = actions[:, 2].long()
        regions, cells = self._position_to_region_cell(y, x)
        active = (cards != 0).float()
        return (
            self.card_distribution.log_prob(cards)
            + active * self._region_distribution(cards).log_prob(regions)
            + active * self._cell_distribution(cards, regions).log_prob(cells)
        )

    def entropy(self):
        region_distribution = torch.distributions.Categorical(
            logits=self.region_logits
        )
        cell_distribution = torch.distributions.Categorical(logits=self.cell_logits)
        conditional_entropy = region_distribution.entropy() + (
            region_distribution.probs * cell_distribution.entropy()
        ).sum(dim=-1)
        return self.card_distribution.entropy() + (
            self.card_distribution.probs[:, 1:] * conditional_entropy[:, 1:]
        ).sum(dim=-1)

    def sample(self):
        cards = self.card_distribution.sample()
        regions = self._region_distribution(cards).sample()
        cells = self._cell_distribution(cards, regions).sample()
        y, x = self._region_cell_to_position(regions, cells)
        active = cards != 0
        return torch.stack((cards, y * active, x * active), dim=1)

    def mode(self):
        cards = self.card_distribution.probs.argmax(dim=1)
        regions = self._region_distribution(cards).probs.argmax(dim=1)
        cells = self._cell_distribution(cards, regions).probs.argmax(dim=1)
        y, x = self._region_cell_to_position(regions, cells)
        active = cards != 0
        return torch.stack((cards, y * active, x * active), dim=1)

    def actions_from_params(self, parameters, deterministic=False):
        self.proba_distribution(parameters)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, parameters):
        actions = self.actions_from_params(parameters)
        return actions, self.log_prob(actions)


class CRAutoregressivePolicy(MultiInputActorCriticPolicy):
    def _build(self, lr_schedule):
        self._build_mlp_extractor()
        self.action_dist = AutoregressiveActionDistribution()
        self.action_net = self.action_dist.proba_distribution_net(
            self.mlp_extractor.latent_dim_pi
        )
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _distribution(self, latent_pi):
        return self.action_dist.proba_distribution(self.action_net(latent_pi))

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._distribution(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, self.value_net(latent_vf), distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._distribution(latent_pi)
        return (
            self.value_net(latent_vf),
            distribution.log_prob(actions),
            distribution.entropy(),
        )

    def get_distribution(self, obs):
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._distribution(latent_pi)


if __name__ == "__main__":
    env = CREnv(opponent_model=random_strategy)
    model = PPO(
        CRAutoregressivePolicy,
        env,
        policy_kwargs={"features_extractor_class": CRFeatureExtractor},
        verbose=1,
        tensorboard_log="./cr_logs_autoregressive/",
        device="cuda",
        seed=0,
    )
    callback = CheckpointCallback(
        save_freq=10_000,
        save_path="./cr_logs_autoregressive/",
        name_prefix="cr_autoregressive",
    )
    try:
        model.learn(total_timesteps=1_000_000, callback=callback)
    finally:
        print("Saving model.")
        model.save("cr_autoregressive")
