from clash_env import ClashRoyaleEnv
env = ClashRoyaleEnv()
obs, _ = env.reset()
agent_deploys = opp_deploys = 0
done = False
while not done:
    obs, r, done, trunc, info = env.step(env.action_space.sample())
p0, p1 = env._battle.players
print(f"P0 elixir: {p0.elixir:.1f}  P1 elixir: {p1.elixir:.1f}")
print(f"Entities on field: {len(env._battle.entities)}")