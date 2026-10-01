"""The PPO policy: pick WAIT or a card, then a tile, both restricted to legal moves.

Observations must include `legal_mask` (4, 32, 18): which tiles each hand slot can
use right now. Sampling, log-probabilities and entropy all use the same mask.
Parameter names match older checkpoints, so `PPO.load` still works on them.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environment import entity_names

EMBED = 8          # entity embedding size inside the board encoder
CARD_EMBED = 16    # card embedding size in the policy heads
TILES = 32 * 18


class BoardEncoder(BaseFeaturesExtractor):
    """CNN over the stacked board frames plus hand, elixir and clock.

    Also keeps the first full-resolution conv activation in `self.spatial`
    (B, 32, 32, 18) for the tile scorer.
    """

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.entity_embedding = nn.Embedding(len(entity_names), EMBED)
        frames = observation_space["grid"].shape[0]
        # 15 raw channels: id -> embedding, type -> one-hot(4), 13 numeric stay.
        in_channels = frames * (13 + EMBED + 4)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_out = self.cnn(torch.zeros(1, in_channels, 32, 18)).shape[1]
        # hand embeddings + elixir + one-hot phase + time left
        self.fc = nn.Linear(cnn_out + 5 * EMBED + 1 + 4 + 1, features_dim)

    def forward(self, obs):
        grid = obs["grid"]                                    # (B, F, 32, 18, 15)
        ids = self.entity_embedding(grid[..., 0].long())
        card_type = F.one_hot(grid[..., 1].long(), num_classes=4).float()
        x = torch.cat([grid[..., 2:], ids, card_type], dim=-1)
        x = x.permute(0, 1, 4, 2, 3).flatten(1, 2).float()   # (B, F*C, 32, 18)
        self.spatial = self.cnn[1](self.cnn[0](x))
        board = self.cnn[2:](self.spatial)
        hand = self.entity_embedding(obs["hand"].long()).flatten(1)
        return torch.relu(self.fc(torch.cat(
            [board, hand, obs["elixir"].float(), obs["phase"].float().flatten(1),
             obs["time_till_next_phase"].float()], dim=1)))


class MaskedSpatialPolicy(MultiInputActorCriticPolicy):
    WAIT_PLAY_ENTROPY_WEIGHT = 0.10
    PLACEMENT_ENTROPY_WEIGHT = 0.25

    def __init__(self, observation_space, action_space, lr_schedule, *args,
                 allow_saving=False, **kwargs):
        # With saving, any hand card may be chosen; unaffordable ones mean "save".
        self.allow_saving = bool(allow_saving)
        kwargs.setdefault("features_extractor_class", BoardEncoder)
        if not kwargs.get("share_features_extractor", True):
            raise ValueError("MaskedSpatialPolicy requires a shared feature extractor")
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        latent = self.global_dim = self.mlp_extractor.latent_dim_pi
        # Replaces SB3's MultiDiscrete head. Keep this registration order: the
        # saved optimizer state is matched to parameters by position.
        self.action_net = nn.Identity()
        self.card_embedding = nn.Embedding(len(entity_names), CARD_EMBED)
        self.card_scorer = nn.Sequential(nn.Linear(latent + CARD_EMBED, 64), nn.Tanh(), nn.Linear(64, 1))
        self.noop_head = nn.Linear(latent, 1)
        self.placement_context = nn.Linear(latent + CARD_EMBED, 32)
        self.tile_scorer = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        # Older checkpoints carry an unused elixir-cost buffer; legal_mask replaced it.
        state_dict = {k: v for k, v in state_dict.items() if k != "card_cost_table"}
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    # --- distributions -------------------------------------------------------

    def _latents(self, obs):
        """Returns (latent_pi, spatial map, latent_vf)."""
        latent_pi, latent_vf = self.mlp_extractor(self.extract_features(obs))
        return latent_pi, self.features_extractor.spatial, latent_vf

    def _card_distribution(self, latent_pi, obs):
        """Categorical over [WAIT, slot 1..4], and the hand's card ids (B, 4)."""
        hand = obs["hand"][:, :4].long()
        state = latent_pi.unsqueeze(1).expand(-1, 4, -1)
        scores = self.card_scorer(torch.cat((state, self.card_embedding(hand)), dim=2)).squeeze(2)
        if not self.allow_saving:
            scores = scores.masked_fill(~self._legal_slots(obs), -1e9)
        return Categorical(logits=torch.cat((self.noop_head(latent_pi), scores), dim=1)), hand

    def _placement_distribution(self, latent_pi, spatial, obs, slot):
        """Categorical over the 576 tiles for hand slot `slot` (0-based, (B,)).

        A slot with no legal tile (unaffordable/saving) gets a single placeholder
        tile, so it contributes zero log-probability and entropy.
        """
        card = obs["hand"][:, :4].long().gather(1, slot[:, None]).squeeze(1)
        context = self.placement_context(torch.cat((latent_pi, self.card_embedding(card)), dim=1))
        logits = self.tile_scorer(spatial + context[:, :, None, None]).flatten(1)
        tiles = obs["legal_mask"].flatten(2).bool()[torch.arange(len(slot), device=slot.device), slot].clone()
        tiles[~tiles.any(dim=1), 0] = True
        return Categorical(logits=logits.masked_fill(~tiles, -1e9))

    @staticmethod
    def _legal_slots(obs):
        return obs["legal_mask"].flatten(2).bool().any(dim=2)

    # --- SB3 interface -------------------------------------------------------

    def _sample_action(self, obs, deterministic=False):
        latent_pi, spatial, latent_vf = self._latents(obs)
        cards, _ = self._card_distribution(latent_pi, obs)
        choice = cards.logits.argmax(1) if deterministic else cards.sample()
        place = self._placement_distribution(latent_pi, spatial, obs, (choice - 1).clamp(min=0))
        tile = place.logits.argmax(1) if deterministic else place.sample()
        play = choice != 0
        tile = torch.where(play, tile, torch.zeros_like(tile))
        log_prob = cards.log_prob(choice) + play.float() * place.log_prob(tile)
        return torch.stack((choice, tile // 18, tile % 18), dim=1), log_prob, latent_vf

    def forward(self, obs, deterministic=False):
        actions, log_prob, latent_vf = self._sample_action(obs, deterministic)
        return actions, self.value_net(latent_vf), log_prob

    def _predict(self, observation, deterministic=False):
        return self._sample_action(observation, deterministic)[0]

    def evaluate_actions(self, obs, actions):
        latent_pi, spatial, latent_vf = self._latents(obs)
        actions = actions.long()
        choice, tile = actions[:, 0], actions[:, 1] * 18 + actions[:, 2]
        cards, hand = self._card_distribution(latent_pi, obs)
        place = self._placement_distribution(latent_pi, spatial, obs, (choice - 1).clamp(min=0))
        log_prob = cards.log_prob(choice) + (choice != 0).float() * place.log_prob(tile)
        entropy = self._conditional_entropy(latent_pi, spatial, cards, obs)
        return self.value_net(latent_vf), log_prob, entropy

    # --- exploration bonus ---------------------------------------------------

    def _conditional_entropy(self, latent_pi, spatial, cards, obs):
        """Separately normalized entropies of WAIT-vs-PLAY, card, and tile choice.

        Card and tile entropy are conditional on playing, so they never reward
        spending. A term is zero where that choice is forced, and each term is
        rescaled so forced states do not dilute it across the batch.
        """
        def rescale(eligible):
            return eligible.float() * eligible.numel() / eligible.sum().clamp(min=1)

        legal = self._legal_slots(obs)
        count = legal.sum(dim=1)
        choices = torch.full_like(count, 4) if self.allow_saving else count
        given_play = Categorical(logits=cards.logits[:, 1:])
        wait_play = Categorical(logits=torch.stack(
            (cards.logits[:, 0], torch.logsumexp(cards.logits[:, 1:], dim=1)), dim=1))

        wait_play_entropy = wait_play.entropy() / np.log(2) * rescale(choices >= 1)
        card_entropy = given_play.entropy() / np.log(4) * rescale(choices >= 2)
        # With saving, weight tiles by deploying cards only, so choosing to save
        # cannot lower the placement bonus.
        weights = (Categorical(logits=cards.logits[:, 1:].masked_fill(~legal, -1e9)).probs
                   if self.allow_saving else given_play.probs)
        placement = torch.zeros_like(card_entropy)
        raw_placement = torch.zeros_like(card_entropy)
        for slot in range(4):
            slots = torch.full_like(count, slot)
            tile_entropy = self._placement_distribution(latent_pi, spatial, obs, slots).entropy()
            placement += weights[:, slot] * tile_entropy
            raw_placement += given_play.probs[:, slot] * tile_entropy
        placement_entropy = placement / np.log(TILES) * rescale(count >= 1)

        objective = (self.WAIT_PLAY_ENTROPY_WEIGHT * wait_play_entropy + card_entropy
                     + self.PLACEMENT_ENTROPY_WEIGHT * placement_entropy)
        self.entropy_diagnostics = {
            "wait_play": wait_play_entropy.mean().detach(),
            "actionable_card": card_entropy.mean().detach(),
            "legal_placement": placement_entropy.mean().detach(),
            "joint_action": (cards.entropy() + (1 - cards.probs[:, 0]) * raw_placement).mean().detach(),
            "objective": objective.mean().detach(),
        }
        return objective
