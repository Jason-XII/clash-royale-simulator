"""Exact simulator legality for the spatial policy, including PPO likelihoods."""
import numpy as np
import torch
from torch.distributions import Categorical
from gymnasium import ObservationWrapper, spaces

from core import Position
from card_utils import Card
from spatial_policy import SpatialPlacementPolicy


class LegalPlacement(ObservationWrapper):
    """Expose the learner's legal actions in its current reflected coordinates."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            **env.observation_space.spaces,
            "legal_mask": spaces.MultiBinary((4, 32, 18)),
        })

    def observation(self, observation):
        battle = self.env.unwrapped.battle
        player = battle.players[0]
        mask = np.zeros((4, 32, 18), dtype=np.int8)
        playable = [player.can_play_card(card) for card in player.cycle[:4]]
        if any(playable):
            troop_tiles = np.array([
                [battle.can_place_troop(0, Position(x + .5, y + .5))
                 for x in range(18)] for y in range(32)
            ], dtype=np.int8)
            for slot, card in enumerate(player.cycle[:4]):
                if playable[slot]:
                    mask[slot] = 1 if Card(card).type == "spell" else troop_tiles
        if self.env.reflected:
            mask = mask[:, :, ::-1].copy()
        return dict(observation, legal_mask=mask)


class MaskedSpatialPolicy(SpatialPlacementPolicy):
    """Use the same action mask for sampling, log probability and entropy."""

    def _latents(self, obs):
        global_and_map, latent_vf = super()._latents(obs)
        # Carry the mask and hand IDs with their corresponding latent batch.
        # PPO minibatches and entropy's per-card queries can then reuse it safely.
        return torch.cat((global_and_map, obs["legal_mask"].flatten(1),
                          obs["hand"][:, :4].float()), dim=1), latent_vf

    def _card_distribution(self, latent_pi, obs):
        distribution, hand = super()._card_distribution(latent_pi, obs)
        legal_cards = obs["legal_mask"].flatten(2).bool().any(dim=2)
        logits = distribution.logits.clone()
        logits[:, 1:] = logits[:, 1:].masked_fill(~legal_cards, -1e9)
        return Categorical(logits=logits), hand

    def _placement_distribution(self, latent_pi, selected_card):
        base_size = self.global_dim + 32 * 32 * 18
        distribution = super()._placement_distribution(latent_pi[:, :base_size], selected_card)
        all_tiles = latent_pi[:, base_size:base_size + 4 * 32 * 18].reshape(-1, 4, 576).bool()
        hand = latent_pi[:, base_size + 4 * 32 * 18:].long()
        slot = (hand == selected_card[:, None]).long().argmax(dim=1)
        tiles = all_tiles[torch.arange(len(slot), device=slot.device), slot].clone()
        empty = ~tiles.any(dim=1)
        # WAIT and unavailable branches need a finite placeholder only for PPO
        # entropy/log-probability calculations. They have zero play probability.
        tiles[empty, 0] = True
        return Categorical(logits=distribution.logits.masked_fill(~tiles, -1e9))

    def _conditional_entropy(self, latent_pi, card_dist, hand, obs):
        # Identical normalized entropy objective to the parent, with simulator
        # actionability replacing float32 elixir comparisons.
        conditional = Categorical(logits=card_dist.logits[:, 1:])
        count = obs["legal_mask"].flatten(2).bool().any(dim=2).sum(dim=1)
        card_eligible = count >= 2
        card_scale = card_eligible.numel() / card_eligible.sum().clamp(min=1)
        card_entropy = (conditional.entropy() / np.log(4)
                        * card_eligible.float() * card_scale)
        placement_entropy = torch.zeros_like(card_entropy)
        for slot in range(4):
            placement = self._placement_distribution(latent_pi, hand[:, slot])
            placement_entropy += conditional.probs[:, slot] * placement.entropy()
        place_eligible = count >= 1
        place_scale = place_eligible.numel() / place_eligible.sum().clamp(min=1)
        placement_entropy = (placement_entropy / np.log(576)
                             * (1 - card_dist.probs[:, 0])
                             * place_eligible.float() * place_scale)
        return card_entropy + self.PLACEMENT_ENTROPY_WEIGHT * placement_entropy
