from card_utils import card_data, Card
import json
import httpx
from pathlib import Path

CARDS = [
    "Knight", "Giant", "Archer", "Goblins", "Pekka", "MiniPekka",
    "Minions", "Skeletons", "SkeletonArmy", "Balloon", "Witch",
    "Barbarians", "Golem", "Valkyrie", "Bomber", "Musketeer",
    "BabyDragon", "Prince", "Wizard", "SpearGoblins",
    "GiantSkeleton", "HogRider", "MinionHorde","RoyalGiant",
    "Princess", "ThreeMusketeers", "BlowdartGoblin", "AngryBarbarians",
    "Bats", "DartBarrell", "RoyalHogs", "Cannon", "Xbow",
    "IceWizard", "SkeletonWarriors", "DarkPrince", "LavaHound",
    "IceSpirits", "FireSpirits", "Miner", "ZapMachine", "Bowler",
    "Rage", "RageBarbarian", "BattleRam", "Fireball", "Arrows"
]

print(len(CARDS))
