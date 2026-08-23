"""
Time how much time pure simulation can speed up compared to real games.
"""
import battle, player
from core import Position
import random, time

player_0_deck = ['Knight', 'MiniPekka', 'Arrows', 'Minions', 'Musketeer', 'Fireball', 'Giant', 'Archer']
player_1_deck = ['Minions', 'Archer', 'MiniPekka', 'Musketeer', 'Giant', 'Fireball', 'Arrows', 'Knight']

b = battle.BattleState(player.PlayerState(0, player_0_deck, 10),
                       player.PlayerState(1, player_1_deck, 10))

t0 = time.time()
while not b.game_over:
    player_id = random.randint(0, 1)
    b.deploy_card(player_id, random.choice(b.players[player_id].cycle[:4]), Position(random.randint(0, 17), random.randint(0,31)))

    for i in range(10):b.step(1/20)
t1 = time.time()
print(t1-t0, b.time/(t1-t0))
