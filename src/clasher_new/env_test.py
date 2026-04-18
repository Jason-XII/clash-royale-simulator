"""smoke_test.py — Quick sanity check for ClashRoyaleEnv"""
import numpy as np
from env import ClashRoyaleEnv, SelfPlayEnv

DECK_A = ["Knight", "Archers", "Giant", "Musketeer",
           "Valkyrie", "Bomber", "Prince", "BabyDragon"]
DECK_B = ["Pekka", "MiniPekka", "Wizard", "Goblins",
           "HogRider", "Witch", "Barbarians", "Minions"]

# ── Single-agent (opponent idles) ────────────────────────────────
env = ClashRoyaleEnv(DECK_A, DECK_B)
obs, info = env.reset()

print("Observation shapes:")
for k, v in obs.items():
    print(f"  {k}: {v.shape}  dtype={v.dtype}")

# Run a few no-op steps
for i in range(10):
    obs, r, done, trunc, info = env.step(env.no_op())
    if done: break
print(f"After 10 no-ops: time={info['time']:.2f}s  reward={r:.4f}  done={done}")

# Try deploying a card
action = {"card": 0, "position": np.array([9.0, 5.0], dtype=np.float32)}
obs, r, done, trunc, info = env.step(action)
print(f"After deploy: elixir={obs['global'][0]*10:.1f}  reward={r:.4f}")

# ── Self-play ────────────────────────────────────────────────────
sp = SelfPlayEnv(DECK_A, DECK_B)
obs0, obs1, info = sp.reset()
noop = ClashRoyaleEnv.no_op()
obs0, obs1, r0, r1, done, info = sp.step(noop, noop)
print(f"Self-play step: r0={r0:.4f}  r1={r1:.4f}  r0+r1={r0+r1:.4f} (should ≈ 0)")

print("\n✅ Smoke test passed!")
