from environment import CREnv, random_strategy, entity_names
from strategies import defensive_strategy, bridge_pressure_strategy, split_lane_strategy, counterpush_strategy

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import torch.nn as nn
import torch.nn.functional as F
import torch

import random
import numpy as np

import os

class CRFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256,
                 embedding_dim: int = 8, channels=(32, 64, 64),
                 normalize_output: bool = False):
        super().__init__(observation_space, features_dim)
        self.embedding_dim = embedding_dim
        self.entity_embedding = nn.Embedding(len(entity_names), self.embedding_dim)
        self.action_embedding = nn.Embedding(len(entity_names), 8)
        self.action_encoder = nn.Sequential(
            nn.Linear(13, 32), nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * 8 * 2, 128), nn.ReLU(),
        )

        self.num_frames = int(observation_space["grid"].shape[0])
        self.per_frame_channels = 13 + self.embedding_dim + 4
        self.in_channels = self.num_frames * self.per_frame_channels

        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, channels[0], 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(channels[1], channels[2], 3, padding=1, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, 32, 18)
            cnn_out = self.cnn(dummy).shape[1]
        self.fc = nn.Linear(cnn_out + 5 * self.embedding_dim + 1 + 4 + 1 + 128, features_dim)
        self.output_norm = nn.LayerNorm(features_dim) if normalize_output else nn.Identity()

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
        action_history = observation["action_history"]
        action_ids = action_history[..., 0].long()
        action_feat = self.action_encoder(torch.cat((
            self.action_embedding(action_ids), action_history[..., 1:]
        ), dim=-1))
        phase = observation["phase"].float().flatten(1)  # (B, 1)
        time_left = observation["time_till_next_phase"].float()
        combined = torch.cat([grid_feat, hand_feat, action_feat, elixir.float(), phase, time_left], dim=1)
        return self.output_norm(torch.relu(self.fc(combined)))


class LargeCRFeatureExtractor(CRFeatureExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(
            observation_space,
            features_dim=features_dim,
            embedding_dim=16,
            channels=(64, 128, 128),
            normalize_output=True,
        )


class SpatialCRFeatureExtractor(CRFeatureExtractor):
    """Compact global encoder that also retains a spatial map for placement."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(
            observation_space,
            features_dim=features_dim,
            embedding_dim=8,
            channels=(32, 64, 64),
            normalize_output=False,
        )
        self.spatial_channels = 32
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
        return self.output_norm(torch.relu(self.fc(combined)))

opponent_pool = [defensive_strategy, bridge_pressure_strategy, split_lane_strategy, counterpush_strategy]

def make_env(rank):
    def factory():
        random.seed(10_000 + rank)
        np.random.seed(10_000 + rank)
        torch.set_num_threads(1)
        return CREnv(opponent_pool=opponent_pool)
    return factory


if __name__ == '__main__':
    debug = False
    if not debug:
        n_envs = 16
        env = SubprocVecEnv([make_env(rank) for rank in range(n_envs)], start_method="spawn")
        env = VecMonitor(env)
        n_steps = 8192 // n_envs
    else:
        env = CREnv(opponent_pool=opponent_pool)
        n_steps = 2048
        n_envs = 1

    model_name = "cr_stacked_moe"

    if (not os.path.exists(f'{model_name}.zip')) or debug:
        print('Previous checkpoint does not existing, training new one from scratch.')
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs={"features_extractor_class": SpatialCRFeatureExtractor},
            n_steps=n_steps,
            # 256 per environment
            batch_size=256,
            learning_rate=1e-4,
            n_epochs=4,
            target_kl=0.03,
            device="cuda",
            seed=0,
            verbose=1,
            tensorboard_log=f"./{model_name}_dir/",
        )
    else:
        model = PPO.load(model_name, env=env, device="cuda", learning_rate=1e-4, n_epochs=4,target_kl=0.03,tensorboard_log=f"./{model_name}_dir/")
    cb = CheckpointCallback(save_freq=20_000 // n_envs, save_path=f"./{model_name}_dir/", name_prefix="cr")
    try:
        model.learn(total_timesteps=15_000_000, reset_num_timesteps=False, callback=[cb])
    finally:
        print('Saving model.')
        model.save(model_name)
