from environment import CREnv, random_strategy
from train import CRAutoregressivePolicy, CRTransformerExtractor
from stable_baselines3 import PPO

policy_kwargs = {
    "features_extractor_class": CRTransformerExtractor,
    "net_arch": {"pi": [], "vf": []},
}

setup_env = CREnv(opponent_model=random_strategy)
blue_model = PPO(CRAutoregressivePolicy, setup_env, policy_kwargs=policy_kwargs)
red_model = PPO(CRAutoregressivePolicy, setup_env, policy_kwargs=policy_kwargs)

env = CREnv(
    opponent_model=lambda obs: red_model.predict(obs, deterministic=False)[0],
    visualize=True,
)

obs, _ = env.reset()
done = False

while not done:
    action, _ = blue_model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(f"Winner: player {env.battle.winner}")