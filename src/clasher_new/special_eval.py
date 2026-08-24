from evaluate import SequentialEvalEnv
from stable_baselines3 import PPO
import torch

env = SequentialEvalEnv(start_deck=['Knight', 'MiniPekka', 'Arrows', 'Giant', 'Musketeer', 'Fireball', 'Minions', 'Archer'],
                        events=[('Giant', 3, 13, 0.5),
                                ('MiniPekka', 3, 12, 0.5)],
                        visualize=True, speed=1)
model = PPO.load(f"cr_logs/cr_5655600_steps.zip")
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs,deterministic=True)
    obs, reward, termination, truncation, info = env.step(action)
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        value = model.policy.predict_values(obs_tensor)
    done = termination or truncation
