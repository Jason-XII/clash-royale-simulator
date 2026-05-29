from environment import CREnv, random_strategy, entity_names
import time
from stable_baselines3 import PPO
from train import CRFeatureExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env():
    return CREnv(opponent_model=lambda obs: random_strategy(obs))

if __name__ == '__main__':

    env = make_env()

    model = PPO('MultiInputPolicy', env, policy_kwargs={"features_extractor_class": CRFeatureExtractor},
                verbose=1, tensorboard_log="./cr_logs")

    # Test env stepping alone (no model)
    t0 = time.time()
    model.learn(1)
    print('Time for 2048 timesteps: ', time.time()-t0)
