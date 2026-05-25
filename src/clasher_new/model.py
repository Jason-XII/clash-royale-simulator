import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environment import entity_names

NUM_CARDS = len(entity_names)
EMBED_DIM = 8

class CRFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.card_embed = nn.Embedding(NUM_CARDS, EMBED_DIM)
        # grid: 15 channels, but channel 0 is the card id -> replace with embedding
        # so the CNN input has (15 - 1) + EMBED_DIM channels
        in_ch = 14 + EMBED_DIM
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        # work out CNN output size with a dry run
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, 32, 18)
            cnn_out = self.cnn(dummy).shape[1]
        self.fc = nn.Linear(cnn_out + 5 * EMBED_DIM + 1, features_dim)

    def forward(self, obs):
        grid = obs['grid']                       # (B, 32, 18, 15)
        hand = obs['hand'].long()                # (B, 5)
        elixir = obs['elixir']                   # (B, 1)

        card_ids = grid[..., 0].long()           # (B, 32, 18)
        card_vecs = self.card_embed(card_ids)    # (B, 32, 18, EMBED)
        rest = grid[..., 1:]                     # (B, 32, 18, 14)
        x = torch.cat([card_vecs, rest], dim=-1) # (B, 32, 18, 14+EMBED)
        x = x.permute(0, 3, 1, 2).float()        # (B, C, 32, 18)
        grid_feat = self.cnn(x)

        hand_feat = self.card_embed(hand).flatten(1)  # (B, 5*EMBED)
        combined = torch.cat([grid_feat, hand_feat, elixir.float()], dim=1)
        return torch.relu(self.fc(combined))


