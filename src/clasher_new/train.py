from environment import CREnv, random_strategy, entity_names

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import torch.nn as nn
import torch.nn.functional as F
import torch

import random
import numpy as np

import os

class CRFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.embedding_dim = 8
        self.entity_embedding = nn.Embedding(len(entity_names), self.embedding_dim)
        self.in_channels = 13 + self.embedding_dim + 4
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, 32, 18)
            cnn_out = self.cnn(dummy).shape[1]
        self.fc = nn.Linear(cnn_out + 5 * self.embedding_dim + 1, features_dim)

    def forward(self, observation):
        """
        Gets the observation, use the embedding (dim=8) to expand the channels, then use one-hot to further expand the channels.
        The code is ugly but should do the work.
        """
        grid = observation['grid']  # (B, 32, 18, 15)
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
        x = x.permute(0, 3, 1, 2).float()  # (B, C, 32, 18)

        grid_feat = self.cnn(x)

        hand_feat = self.entity_embedding(hand).flatten(1)  # (B, 5*EMBED)
        combined = torch.cat([grid_feat, hand_feat, elixir.float()], dim=1)
        return torch.relu(self.fc(combined))


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


def make_env(rank):
    def factory():
        random.seed(10_000 + rank)
        np.random.seed(10_000 + rank)
        torch.set_num_threads(1)
        model_names = os.listdir('opponent_pool')
        models = [PPO.load(f'opponent_pool/{name}') for name in model_names]
        model_funcs = [lambda o: m.predict(o)[0] for m in models]
        # return CREnv(opponent_model=lambda o: m.predict(o)[0])
        return CREnv(opponent_pool=model_funcs)
    return factory


if __name__ == '__main__':
    n_envs = 16
    env = SubprocVecEnv([make_env(rank) for rank in range(n_envs)], start_method="spawn")
    env = VecMonitor(env)
    n_steps = 8192 // n_envs

    if not os.path.exists('cr_random_new.zip'):
        print('Previous checkpoint does not exisiting, training new one from scratch.')
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs={"features_extractor_class": CRFeatureExtractor},
            n_steps=n_steps,
            # 256 per environment
            batch_size=256,
            learning_rate=1e-4,
            n_epochs=4,
            target_kl=0.03,
            device="cuda",
            seed=0,
            verbose=1,
            tensorboard_log="./cr_randomnew/",
        )
    else:
        model = PPO.load("cr_random_new", env=env, device="cuda", learning_rate=1e-4, n_epochs=4,target_kl=0.03,ent_coef = 0.001,tensorboard_log="./cr_randomnew/")
    cb = CheckpointCallback(save_freq=20_000 // n_envs, save_path="./cr_randomnew/", name_prefix="cr")
    try:
        model.learn(total_timesteps=5_000_000, reset_num_timesteps=False, callback=[cb])
    finally:
        print('Saving model.')
        model.save('cr_random_new')
