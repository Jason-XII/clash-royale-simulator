from environment import CREnv, random_strategy, entity_names
import time
from stable_baselines3 import PPO
from train import CRFeatureExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env():
    return CREnv(opponent_model=lambda obs: random_strategy(obs))

if __name__ == '__main__':

    env = make_env()
    env.reset()

    model = PPO('MultiInputPolicy', env, policy_kwargs={"features_extractor_class": CRFeatureExtractor},
                verbose=1, tensorboard_log="./cr_logs")
    model.save('cr_discrete')
    print('Stepping the environment 100 times:')
    t0 = time.time()
    for i in range(100): env.step(random_strategy([]))
    t1 = time.time()
    print('Took', (t1-t0)*10, 'milliseconds per step.')
