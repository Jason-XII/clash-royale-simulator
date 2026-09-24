"""Local tile scoring with the baseline encoder, card policy and PPO entropy."""
import torch
from torch import nn
from torch.distributions import Categorical

from train import CRFeatureExtractor
from train_autoregressive import ContentMaskedAutoregressivePolicy


class RetainedMapCNN(nn.Sequential):
    """Keep the first full-resolution activation without changing global output."""

    def forward(self, x):
        x = self[1](self[0](x))
        self.spatial = x
        for layer in list(self)[2:]:
            x = layer(x)
        return x


class SpatialFeatures(CRFeatureExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.cnn = RetainedMapCNN(*self.cnn)


class SpatialPlacementPolicy(ContentMaskedAutoregressivePolicy):
    """Replace the 576-output placement MLP with one shared convolutional scorer.

    No new masks, observations, coordinates, or entropy terms. The global actor
    still sees position through the baseline encoder; this is not a guarantee
    of reflection equivariance or immunity to learning absolute tile habits.
    """

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        kwargs.setdefault("features_extractor_class", SpatialFeatures)
        if not kwargs.get("share_features_extractor", True):
            raise ValueError("Spatial placement requires a shared feature extractor")
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        if not isinstance(self.features_extractor, SpatialFeatures):
            raise ValueError("SpatialPlacementPolicy requires SpatialFeatures")
        self.global_dim = self.mlp_extractor.latent_dim_pi
        # Remove unused baseline placement parameters from both model/optimizer.
        del self.placement_net
        del self.placement_head
        self.placement_context = nn.Linear(self.global_dim + 16, 32)
        self.tile_scorer = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _latents(self, obs):
        global_pi, global_vf = super()._latents(obs)
        spatial = self.features_extractor.cnn.spatial
        # Carry the map with its latent instead of reading mutable cached state
        # later. Repeated/interleaved batches in evaluation stay correctly aligned.
        return torch.cat((global_pi, spatial.flatten(1)), dim=1), global_vf

    def _card_distribution(self, latent_pi, obs):
        return super()._card_distribution(latent_pi[:, :self.global_dim], obs)

    def _placement_distribution(self, latent_pi, selected_card):
        global_pi = latent_pi[:, :self.global_dim]
        spatial = latent_pi[:, self.global_dim:].reshape(-1, 32, 32, 18)
        context = self.placement_context(torch.cat(
            (global_pi, self.card_embedding(selected_card)), dim=1
        ))[:, :, None, None]
        logits = self.tile_scorer(spatial + context).flatten(1)
        return Categorical(logits=logits)
