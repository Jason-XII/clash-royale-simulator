"""bench_sim.py — Benchmark BattleState.step() throughput"""
import time
import random
from battle import BattleState
from player import PlayerState
from core import Position

DECK_A = ["Knight", "Archer", "Giant", "Musketeer",
           "Valkyrie", "Bomber", "Prince", "BabyDragon"]
DECK_B = ["Pekka", "MiniPekka", "Wizard", "Goblins",
           "HogRider", "Witch", "Barbarians", "Minions"]

p0 = PlayerState(0, list(DECK_A), 5.0)
p1 = PlayerState(1, list(DECK_B), 5.0)
battle = BattleState(p0, p1)

# Deploy a few cards to make it realistic
battle.deploy_card(0, "Knight",    Position(9.5, 5.5))
battle.deploy_card(0, "Archers",   Position(7.5, 6.5))
battle.deploy_card(1, "Pekka",     Position(9.5, 26.5))
battle.deploy_card(1, "Goblins",   Position(11.5, 25.5))

# Warm up
for _ in range(100):
    battle.step(battle.dt)

# Benchmark single ticks
N = 6000  # ~100 seconds of game time
t0 = time.perf_counter()
for _ in range(N):
    battle.step(battle.dt)
t1 = time.perf_counter()

tick_us = (t1 - t0) / N * 1e6
decision_ms = tick_us * 20 / 1000  # 20 ticks per decision

print(f"Entities alive: {sum(1 for e in battle.entities.values() if e.is_alive)}")
print(f"Single tick:    {tick_us:.1f} µs")
print(f"Per decision:   {decision_ms:.2f} ms  (20 ticks)")
print(f"Realtime ratio: {1/60 / (tick_us/1e6):.0f}x faster than realtime")
print(f"Full game est:  {tick_us * 180 * 60 / 1e6:.1f}s  (for 3min game)")
