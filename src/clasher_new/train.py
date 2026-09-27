from environment import entity_names
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch.nn.functional as F
import torch

class CRFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.embedding_dim = 8
        self.entity_embedding = nn.Embedding(len(entity_names), self.embedding_dim)

        self.num_frames = int(observation_space["grid"].shape[0])
        self.per_frame_channels = 13 + self.embedding_dim + 4
        self.in_channels = self.num_frames * self.per_frame_channels

        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, 32, 18)
            cnn_out = self.cnn(dummy).shape[1]
        self.fc = nn.Linear(cnn_out + 5 * self.embedding_dim + 1 + 4 + 1, features_dim)

    def forward(self, observation):
        """
        Gets the observation, use the embedding (dim=8) to expand the channels, then use one-hot to further expand the channels.
        The code is ugly but should do the work.
        """
        grid = observation['grid']  # (B, 8, 32, 18, 15)
        hand = observation['hand'].long()  # (B, 5)
        elixir = observation['elixir']

        card_ids = grid[..., 0].long()
        card_vecs = self.entity_embedding(card_ids)

        rest = grid[..., 1:]  # (B, 32, 18, 14)
        x = torch.cat([rest, card_vecs], dim=-1)  # (B, 32, 18, 14+EMBED)
        card_type = x[..., 0].long()  # (B, 32, 18)
        card_type_oh = F.one_hot(card_type, num_classes=4).float()  # (B, 32, 18, 4)
        rest = x[..., 1:]
        x = torch.cat([rest, card_type_oh], dim=-1)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.flatten(1, 2).float()

        grid_feat = self.cnn(x)

        hand_feat = self.entity_embedding(hand).flatten(1)  # (B, 5*EMBED)
        phase = observation["phase"].float().flatten(1)  # (B, 1)
        time_left = observation["time_till_next_phase"].float()
        combined = torch.cat([grid_feat, hand_feat, elixir.float(), phase, time_left], dim=1)
        return torch.relu(self.fc(combined))
