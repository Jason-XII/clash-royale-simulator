from environment import CREnv, random_strategy, entity_names

from gymnasium import spaces, Wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from train import CRFeatureExtractor
from random import randint

class FlatActionWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(1 + 4 * 32 * 18)  # 2305

    def _decode(self, action):
        if action == 0:
            return 0, 0, 0
        action -= 1
        slot = action // (32 * 18) + 1
        pos = action % (32 * 18)
        y = pos // 18
        x = pos % 18
        return slot, y, x

    def step(self, action):
        return self.env.step(self._decode(action))



if __name__ == '__main__':
    env = FlatActionWrapper(CREnv(opponent_model=lambda obs: random_strategy(obs)))
    model = PPO(
        "MultiInputPolicy", env,
        policy_kwargs={"features_extractor_class": CRFeatureExtractor},
        verbose=1, tensorboard_log="./cr_logs_flat/", device="cuda", seed=0,
    )
    cb = CheckpointCallback(save_freq=10_000, save_path="./cr_logs/", name_prefix="cr_flat")
    try:
        model.learn(total_timesteps=1_000_000, reset_num_timesteps=False, callback=[cb])
    finally:
        print('Saving model.')
        model.save('cr_flat')


