from environment import CREnv, random_strategy, entity_names
import time
from stable_baselines3 import PPO
from train import CRFeatureExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env():
    return CREnv(opponent_model=lambda obs: random_strategy(obs))

if __name__ == '__main__':
    env = make_env()
    t0 = time.time()
    t_irl = 0
    for i in range(100):
        state, _ = env.reset()
        running = True
        # model = PPO('MultiInputPolicy', env, policy_kwargs={"features_extractor_class": CRFeatureExtractor},
        #             verbose=1, tensorboard_log="./cr_logs")
        # model.save('cr_discrete')
        while running:
            state, reward, termination, truncation, info = env.step(random_strategy(state))
            running = not (termination or truncation)
        print('game')
        t_irl += env.battle.time
    t1 = time.time()
    print( t_irl, t1-t0, f'\n The simulator is {t_irl/(t1-t0)}x faster than real time.')
