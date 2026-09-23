"""Spatial encoder and masked card-then-tile PPO policy."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from gymnasium import spaces
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environment import entity_names


class SpatialEncoder(BaseFeaturesExtractor):
    """Encode global state while retaining a tile-aligned placement map."""

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.embedding_dim = 8
        self.spatial_channels = 32
        self.in_channels = int(observation_space["grid"].shape[0]) * (13 + 8 + 4)
        self.entity_embedding = nn.Embedding(len(entity_names), self.embedding_dim)
        self.action_embedding = nn.Embedding(len(entity_names), 8)
        self.action_encoder = nn.Sequential(
            nn.Linear(13, 32), nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * 8 * 2, 128), nn.ReLU(),
        )
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.global_cnn = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, 32, 18)
            cnn_out = self.global_cnn(self.spatial_cnn(dummy)).shape[1]
        self.fc = nn.Linear(cnn_out + 5 * self.embedding_dim + 1 + 4 + 1 + 128, features_dim)
        self.spatial_features = None

    def forward(self, observation):
        grid = observation["grid"]
        hand = observation["hand"].long()
        card_ids = grid[..., 0].long()
        card_vecs = self.entity_embedding(card_ids)
        rest = grid[..., 1:]
        x = torch.cat((rest, card_vecs), dim=-1)
        card_type = x[..., 0].long()
        card_type_oh = F.one_hot(card_type, num_classes=4).float()
        x = torch.cat((x[..., 1:], card_type_oh), dim=-1)
        x = x.permute(0, 1, 4, 2, 3).contiguous().flatten(1, 2).float()

        self.spatial_features = self.spatial_cnn(x)
        grid_feat = self.global_cnn(self.spatial_features)
        hand_feat = self.entity_embedding(hand).flatten(1)
        action_history = observation["action_history"]
        action_ids = action_history[..., 0].long()
        action_feat = self.action_encoder(torch.cat((
            self.action_embedding(action_ids), action_history[..., 1:]
        ), dim=-1))
        phase = observation["phase"].float().flatten(1)
        combined = torch.cat((
            grid_feat,
            hand_feat,
            action_feat,
            observation["elixir"].float(),
            phase,
            observation["time_till_next_phase"].float(),
        ), dim=1)
        return torch.relu(self.fc(combined))


class ClashPolicy(MultiInputActorCriticPolicy):
    """Score hand cards with legal placements, then place the selected card."""

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

    def __init__(self, observation_space, action_space, lr_schedule, *args,
                 card_embedding_dim=16, hidden_dim=64, **kwargs):
        expected_nvec = np.array([5, 32, 18])
        if not isinstance(action_space, spaces.MultiDiscrete) or not np.array_equal(
            action_space.nvec, expected_nvec
        ):
            raise ValueError(
                "ClashPolicy requires MultiDiscrete([5, 32, 18])."
            )

        kwargs.setdefault("features_extractor_class", SpatialEncoder)
        if kwargs.get("share_features_extractor", True) is not True:
            raise ValueError("ClashPolicy requires a shared spatial encoder")
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        if not isinstance(self.features_extractor, SpatialEncoder):
            raise ValueError("ClashPolicy requires SpatialEncoder")

        latent_dim = self.mlp_extractor.latent_dim_pi
        self.action_net = nn.Identity()
        self.card_embedding = nn.Embedding(len(entity_names), card_embedding_dim)
        self.card_scorer = nn.Sequential(
            nn.Linear(latent_dim + card_embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.noop_head = nn.Linear(latent_dim, 1)
        self.spatial_channels = self.features_extractor.spatial_channels
        self.placement_condition = nn.Linear(
            latent_dim + card_embedding_dim, 2 * self.spatial_channels
        )
        y_coordinates = torch.linspace(-1.0, 1.0, 32).view(1, 1, 32, 1)
        x_coordinates = torch.linspace(-1.0, 1.0, 18).view(1, 1, 1, 18)
        coordinates = torch.cat((
            y_coordinates.expand(1, 1, 32, 18),
            x_coordinates.expand(1, 1, 32, 18),
        ), dim=1)
        self.register_buffer("placement_coordinates", coordinates)
        self.placement_conv = nn.Sequential(
            nn.Conv2d(self.spatial_channels + 2, self.spatial_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.spatial_channels, 1, 1),
        )

        cost_table = torch.full((len(entity_names),), float("inf"))
        cost_table[0] = 0.0
        for card_name, cost in self.CARD_COSTS.items():
            cost_table[entity_names.index(card_name)] = float(cost)
        # Retain this buffer for checkpoint compatibility; legality comes from
        # the simulator's masks, not a second calculation using rounded elixir.
        self.register_buffer("card_cost_table", cost_table)

        # The parent created its optimizer before these new heads existed.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _latents(self, obs):
        return self.mlp_extractor(self.extract_features(obs))

    @staticmethod
    def _playable_slots(obs):
        return obs["placement_mask"][:, 1:].flatten(2).bool().any(dim=-1)

    def _card_distribution(self, latent_pi, obs):
        hand = obs["hand"][:, :4].long()
        card_vectors = self.card_embedding(hand)
        state = latent_pi.unsqueeze(1).expand(-1, hand.shape[1], -1)
        card_scores = self.card_scorer(
            torch.cat((state, card_vectors), dim=2)
        ).squeeze(2)

        card_scores = card_scores.masked_fill(~self._playable_slots(obs), -1e9)

        noop_score = self.noop_head(latent_pi)
        logits = torch.cat((noop_score, card_scores), dim=1)
        return Categorical(logits=logits), hand

    @staticmethod
    def _selected_card_ids(hand, slot):
        hand_index = (slot - 1).clamp(0, 3)
        selected_card = hand.gather(1, hand_index.unsqueeze(1)).squeeze(1)
        return torch.where(slot == 0, torch.zeros_like(selected_card), selected_card)

    def _placement_distribution(self, latent_pi, selected_card, placement_mask):
        conditioned = torch.cat(
            (latent_pi, self.card_embedding(selected_card)), dim=1
        )
        spatial = self.features_extractor.spatial_features
        if spatial is None or spatial.shape[0] != latent_pi.shape[0]:
            raise RuntimeError("spatial features must be encoded before placement")
        scale, bias = self.placement_condition(conditioned).chunk(2, dim=1)
        scale = torch.tanh(scale).unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)
        placement = spatial * (1.0 + scale) + bias
        coordinates = self.placement_coordinates.expand(
            placement.shape[0], -1, -1, -1
        )
        placement = torch.cat((placement, coordinates), dim=1)
        logits = self.placement_conv(placement).flatten(1)
        flat_mask = placement_mask.reshape(placement_mask.shape[0], -1).bool().clone()
        # Unplayable slots have zero card probability, but entropy evaluates
        # every slot. Give only those empty branches one zero-entropy placeholder.
        flat_mask[:, 0] |= ~flat_mask.any(dim=1)
        logits = logits.masked_fill(~flat_mask, -1e9)
        return Categorical(logits=logits)

    def _sample_action(self, latent_pi, obs, deterministic=False):
        card_dist, hand = self._card_distribution(latent_pi, obs)
        slot = torch.argmax(card_dist.logits, dim=1) if deterministic else card_dist.sample()
        selected_card = self._selected_card_ids(hand, slot)
        placement_mask = obs["placement_mask"].gather(
            1, slot[:, None, None, None].expand(-1, 1, 32, 18)
        ).squeeze(1)
        placement_dist = self._placement_distribution(
            latent_pi, selected_card, placement_mask
        )

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

    def _placement_entropies(self, latent_pi, hand, obs):
        """Per-slot tile entropy in nats, with the same masks as sampling."""
        entropies = []
        for slot_index in range(hand.shape[1]):
            placement_dist = self._placement_distribution(
                latent_pi,
                hand[:, slot_index],
                obs["placement_mask"][:, slot_index + 1],
            )
            entropies.append(placement_dist.entropy())
        return torch.stack(entropies, dim=1)

    def _conditional_entropy(self, latent_pi, card_dist, hand, obs):
        """Exact H(slot) + sum_slot P(slot) H(tile | slot), with one WAIT."""
        placement_entropies = self._placement_entropies(latent_pi, hand, obs)
        return card_dist.entropy() + (
            card_dist.probs[:, 1:] * placement_entropies
        ).sum(dim=1)

    @torch.no_grad()
    def exploration_statistics(self, obs):
        """Per-state diagnostics; conditional metrics exclude forced waits."""
        latent_pi, _ = self._latents(obs)
        card_dist, hand = self._card_distribution(latent_pi, obs)
        placement_entropies = self._placement_entropies(latent_pi, hand, obs)
        playable = self._playable_slots(obs)
        actionable = playable.any(dim=1)
        # Normalize logits directly, even when P(play) is extremely small.
        played_card_dist = Categorical(logits=card_dist.logits[:, 1:])
        wait_play_dist = Categorical(logits=torch.stack((
            card_dist.logits[:, 0], torch.logsumexp(card_dist.logits[:, 1:], dim=1),
        ), dim=1))
        slot_entropy = card_dist.entropy()
        return {
            "hand": hand,
            "playable": playable,
            "actionable": actionable,
            "slot_probabilities": card_dist.probs,
            "wait_probability": card_dist.probs[:, 0],
            "slot_entropy": slot_entropy,
            "wait_play_entropy": wait_play_dist.entropy(),
            "joint_entropy": slot_entropy + (
                card_dist.probs[:, 1:] * placement_entropies
            ).sum(dim=1),
            "card_entropy_given_play": torch.where(
                actionable, played_card_dist.entropy(), 0.0,
            ),
            "placement_entropy_given_play": torch.where(
                actionable, (played_card_dist.probs * placement_entropies).sum(dim=1), 0.0,
            ),
            "placement_entropies": placement_entropies,
        }

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
        placement_mask = obs["placement_mask"].gather(
            1, slot[:, None, None, None].expand(-1, 1, 32, 18)
        ).squeeze(1)
        placement_dist = self._placement_distribution(
            latent_pi, selected_card, placement_mask
        )
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
