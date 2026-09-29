import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from torch.distributions import Categorical

from environment import entity_names
from train import CRFeatureExtractor


class AutoregressivePolicy(MultiInputActorCriticPolicy):
    """Choose a card first, then choose a placement conditioned on it."""

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        expected_nvec = np.array([5, 32, 18])
        if not isinstance(action_space, spaces.MultiDiscrete) or not np.array_equal(
            action_space.nvec, expected_nvec
        ):
            raise ValueError("AutoregressivePolicy requires MultiDiscrete([5, 32, 18]).")

        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        latent_dim = self.mlp_extractor.latent_dim_pi
        card_embedding_dim = 16
        placement_hidden_dim = 64

        # Replace SB3's independent MultiDiscrete action head.
        self.action_net = nn.Identity()
        self.card_head = nn.Linear(latent_dim, 5)
        self.card_embedding = nn.Embedding(5, card_embedding_dim)
        self.placement_net = nn.Sequential(
            nn.Linear(latent_dim + card_embedding_dim, placement_hidden_dim),
            nn.Tanh(),
        )
        self.y_head = nn.Linear(placement_hidden_dim, 32)
        self.x_head = nn.Linear(placement_hidden_dim, 18)

        # The parent created its optimizer before these new heads existed.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _latents(self, obs):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            return self.mlp_extractor(features)

        policy_features, value_features = features
        latent_pi = self.mlp_extractor.forward_actor(policy_features)
        latent_vf = self.mlp_extractor.forward_critic(value_features)
        return latent_pi, latent_vf

    def _placement_distributions(self, latent_pi, card):
        conditioned = torch.cat((latent_pi, self.card_embedding(card)), dim=1)
        placement_latent = self.placement_net(conditioned)
        y_dist = Categorical(logits=self.y_head(placement_latent))
        x_dist = Categorical(logits=self.x_head(placement_latent))
        return y_dist, x_dist

    def _sample_action(self, latent_pi, deterministic=False):
        card_dist = Categorical(logits=self.card_head(latent_pi))
        card = torch.argmax(card_dist.logits, dim=1) if deterministic else card_dist.sample()
        y_dist, x_dist = self._placement_distributions(latent_pi, card)
        if deterministic:
            y = torch.argmax(y_dist.logits, dim=1)
            x = torch.argmax(x_dist.logits, dim=1)
        else:
            y = y_dist.sample()
            x = x_dist.sample()

        # A no-op has no meaningful placement, so give it one canonical form.
        play_card = card != 0
        y = torch.where(play_card, y, torch.zeros_like(y))
        x = torch.where(play_card, x, torch.zeros_like(x))
        log_prob = card_dist.log_prob(card)
        log_prob = log_prob + play_card.float() * (
            y_dist.log_prob(y) + x_dist.log_prob(x)
        )
        return torch.stack((card, y, x), dim=1), log_prob

    def _conditional_entropy(self, latent_pi, card_dist):
        """Compute H(card) + E_card[H(position | card)] exactly."""
        entropy = card_dist.entropy()
        card_probabilities = card_dist.probs
        for card_id in range(1, 5):
            card = torch.full(
                (latent_pi.shape[0],), card_id, dtype=torch.long, device=latent_pi.device
            )
            y_dist, x_dist = self._placement_distributions(latent_pi, card)
            entropy = entropy + card_probabilities[:, card_id] * (
                y_dist.entropy() + x_dist.entropy()
            )
        return entropy

    def forward(self, obs, deterministic=False):
        latent_pi, latent_vf = self._latents(obs)
        values = self.value_net(latent_vf)
        actions, log_prob = self._sample_action(latent_pi, deterministic)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        latent_pi, latent_vf = self._latents(obs)
        values = self.value_net(latent_vf)

        actions = actions.long()
        card, y, x = actions[:, 0], actions[:, 1], actions[:, 2]
        card_dist = Categorical(logits=self.card_head(latent_pi))
        y_dist, x_dist = self._placement_distributions(latent_pi, card)
        play_card = card != 0
        log_prob = card_dist.log_prob(card)
        log_prob = log_prob + play_card.float() * (
            y_dist.log_prob(y) + x_dist.log_prob(x)
        )
        entropy = self._conditional_entropy(latent_pi, card_dist)
        return values, log_prob, entropy

    def _predict(self, observation, deterministic=False):
        latent_pi, _ = self._latents(observation)
        actions, _ = self._sample_action(latent_pi, deterministic)
        return actions


class ContentMaskedAutoregressivePolicy(MultiInputActorCriticPolicy):
    """Score actual hand cards, mask unaffordable ones, then place the selected card."""

    WAIT_PLAY_ENTROPY_WEIGHT = 0.10
    PLACEMENT_ENTROPY_WEIGHT = 0.25

    CARD_COSTS = {
        "Knight": 3,
        "MiniPekka": 4,
        "Arrows": 3,
        "Minions": 3,
        "Archer": 3,
        "Musketeer": 4,
        "Fireball": 4,
        "Giant": 5,
    }

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        expected_nvec = np.array([5, 32, 18])
        if not isinstance(action_space, spaces.MultiDiscrete) or not np.array_equal(
            action_space.nvec, expected_nvec
        ):
            raise ValueError(
                "ContentMaskedAutoregressivePolicy requires MultiDiscrete([5, 32, 18])."
            )

        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        latent_dim = self.mlp_extractor.latent_dim_pi
        card_embedding_dim = 16
        hidden_dim = 64

        self.action_net = nn.Identity()
        self.card_embedding = nn.Embedding(len(entity_names), card_embedding_dim)
        self.card_scorer = nn.Sequential(
            nn.Linear(latent_dim + card_embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.noop_head = nn.Linear(latent_dim, 1)
        self.placement_net = nn.Sequential(
            nn.Linear(latent_dim + card_embedding_dim, hidden_dim),
            nn.Tanh(),
        )
        self.placement_head = nn.Linear(hidden_dim, 32 * 18)

        cost_table = torch.full((len(entity_names),), float("inf"))
        cost_table[0] = 0.0
        for card_name, cost in self.CARD_COSTS.items():
            cost_table[entity_names.index(card_name)] = float(cost)
        self.register_buffer("card_cost_table", cost_table)

        # The parent created its optimizer before these new heads existed.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _latents(self, obs):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            return self.mlp_extractor(features)

        policy_features, value_features = features
        latent_pi = self.mlp_extractor.forward_actor(policy_features)
        latent_vf = self.mlp_extractor.forward_critic(value_features)
        return latent_pi, latent_vf

    def _card_distribution(self, latent_pi, obs):
        hand = obs["hand"][:, :4].long()
        card_vectors = self.card_embedding(hand)
        state = latent_pi.unsqueeze(1).expand(-1, hand.shape[1], -1)
        card_scores = self.card_scorer(
            torch.cat((state, card_vectors), dim=2)
        ).squeeze(2)

        elixir = obs["elixir"].float().reshape(-1, 1)
        costs = self.card_cost_table[hand]
        affordable = costs <= elixir
        card_scores = card_scores.masked_fill(~affordable, -1e9)

        noop_score = self.noop_head(latent_pi)
        logits = torch.cat((noop_score, card_scores), dim=1)
        return Categorical(logits=logits), hand

    @staticmethod
    def _selected_card_ids(hand, slot):
        hand_index = (slot - 1).clamp(0, 3)
        selected_card = hand.gather(1, hand_index.unsqueeze(1)).squeeze(1)
        return torch.where(slot == 0, torch.zeros_like(selected_card), selected_card)

    def _placement_distribution(self, latent_pi, selected_card):
        conditioned = torch.cat(
            (latent_pi, self.card_embedding(selected_card)), dim=1
        )
        placement_latent = self.placement_net(conditioned)
        return Categorical(logits=self.placement_head(placement_latent))

    def _sample_action(self, latent_pi, obs, deterministic=False):
        card_dist, hand = self._card_distribution(latent_pi, obs)
        slot = torch.argmax(card_dist.logits, dim=1) if deterministic else card_dist.sample()
        selected_card = self._selected_card_ids(hand, slot)
        placement_dist = self._placement_distribution(latent_pi, selected_card)

        if deterministic:
            tile = torch.argmax(placement_dist.logits, dim=1)
        else:
            tile = placement_dist.sample()

        y = tile // 18
        x = tile % 18

        play_card = slot != 0
        y = torch.where(play_card, y, torch.zeros_like(y))
        x = torch.where(play_card, x, torch.zeros_like(x))
        log_prob = card_dist.log_prob(slot)
        log_prob = log_prob + play_card.float() * placement_dist.log_prob(tile)
        return torch.stack((slot, y, x), dim=1), log_prob

    def _conditional_entropy(self, latent_pi, card_dist, hand, obs):
        """Balanced entropy for wait/play, card, and placement decisions."""
        card_distribution = Categorical(logits=card_dist.logits[:, 1:])
        elixir = obs["elixir"].float().reshape(-1, 1)
        affordable_count = (self.card_cost_table[hand] <= elixir).sum(dim=1)

        wait_play_distribution = Categorical(logits=torch.stack((
            card_dist.logits[:, 0],
            torch.logsumexp(card_dist.logits[:, 1:], dim=1),
        ), dim=1))
        wait_play_eligible = affordable_count >= 1
        wait_play_scale = (
            wait_play_eligible.numel() / wait_play_eligible.sum().clamp(min=1)
        )
        wait_play_entropy = (
            wait_play_distribution.entropy() / np.log(2)
            * wait_play_eligible.float()
            * wait_play_scale
        )

        card_eligible = affordable_count >= 2
        # PPO averages over the whole batch. Rescale so forced waits and states
        # with only one legal card do not dilute the exploration bonus.
        card_scale = card_eligible.numel() / card_eligible.sum().clamp(min=1)
        card_entropy = card_distribution.entropy() / np.log(4)
        card_entropy = card_entropy * card_eligible.float() * card_scale

        # Compute E[H(tile | card)] conditional on playing. Keeping this term
        # independent of P(play) prevents placement diversity from rewarding
        # immediate spending.
        placement_entropy = torch.zeros_like(card_entropy)
        for slot_index in range(hand.shape[1]):
            placement_dist = self._placement_distribution(
                latent_pi, hand[:, slot_index]
            )
            placement_entropy = placement_entropy + (
                card_distribution.probs[:, slot_index] * placement_dist.entropy()
            )

        raw_placement_entropy = placement_entropy
        placement_eligible = affordable_count >= 1
        placement_scale = (
            placement_eligible.numel() / placement_eligible.sum().clamp(min=1)
        )
        placement_entropy = placement_entropy / np.log(32 * 18)
        placement_entropy = (
            placement_entropy
            * placement_eligible.float()
            * placement_scale
        )
        objective = (
            self.WAIT_PLAY_ENTROPY_WEIGHT * wait_play_entropy
            + card_entropy
            + self.PLACEMENT_ENTROPY_WEIGHT * placement_entropy
        )
        self.entropy_diagnostics = {
            "wait_play": wait_play_entropy.mean().detach(),
            "actionable_card": card_entropy.mean().detach(),
            "legal_placement": placement_entropy.mean().detach(),
            "joint_action": (
                card_dist.entropy()
                + (1 - card_dist.probs[:, 0]) * raw_placement_entropy
            ).mean().detach(),
            "objective": objective.mean().detach(),
        }
        return objective

    def forward(self, obs, deterministic=False):
        latent_pi, latent_vf = self._latents(obs)
        values = self.value_net(latent_vf)
        actions, log_prob = self._sample_action(latent_pi, obs, deterministic)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        latent_pi, latent_vf = self._latents(obs)
        values = self.value_net(latent_vf)

        actions = actions.long()
        slot, y, x = actions[:, 0], actions[:, 1], actions[:, 2]
        card_dist, hand = self._card_distribution(latent_pi, obs)
        selected_card = self._selected_card_ids(hand, slot)
        placement_dist = self._placement_distribution(latent_pi, selected_card)
        tile = y * 18 + x
        play_card = slot != 0
        log_prob = card_dist.log_prob(slot)
        log_prob = log_prob + play_card.float() * placement_dist.log_prob(tile)
        entropy = self._conditional_entropy(latent_pi, card_dist, hand, obs)
        return values, log_prob, entropy

    def _predict(self, observation, deterministic=False):
        latent_pi, _ = self._latents(observation)
        actions, _ = self._sample_action(latent_pi, observation, deterministic)
        return actions
