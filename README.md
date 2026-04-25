This project is still under active development. 

# Clash Royale Simulator 

![demo](./demo.gif)

## Requirements

```bash
pip install pygame fastcore --user --no-cache-dir
```

## How to play

Yes, this simulator is already playable! You need two computers that connect to the same local network. 

1. Run `ipconfig` on Windows or `ifconfig | grep inet` on macOS. Find your ip address in the local network. 
If you are at home, the address should look like 192.168.xx.xx, if you are connected to a school network, 
then that will probably start with a 10.xx.xx.xx.

2. Modify the parameter in `server.py` in `src/clasher_new` and run it. Code files all live in this directory. 
3. Send the `client_side` folder to another device and run the `client.py`. 
4. Follow the instructions in the GUI and enter the server ip address after choosing your deck.
5. When two clients are connected to the same server, the game starts automatically! 
You can see elixir, drag and drop to deploy cards, see entities on the screen, etc.

## Why I created this repository
I like clash royale and want to train a RL agent to play the game. 
However, a simulator is needed to speed up training. I searched on GitHub and the only usable project I found
was samdickson22's [repository](https://github.com/samdickson22/clash-simulator). 

I did some investigation and found that although the simulator has most clash royale's features
implemented, the code is almost completely written by AI - hard to read and impossible for humans to make improvements on the code.

After examining the project structure more closely, I spent about 10 hours refactoring the core pieces of the codebase.
In the end, I realized that the only way to make all of this work is to re-implement the whole game from scratch, without any vibe-coding.

## Current Progress

- tower placement: done.
- troop spawning: done.
- pathfinding and going around obstacles: done.
- attacking and following enemy troops in sight: done.
- creating projectiles: done.
- handling entity collision: done.
- periodic spawning, death damage, death spawning
- charging, jumping across rivers
- special slow and freeze effects

Caveats:
- spells are mostly not implemented yet.
- Miner's digging logic needs further refinement
- evo, hero and champion card mechanics are not implemented
- ...

### Characters implemented

- Knight
- Giant
- Archers
- Goblins
- Pekka
- MiniPekka
- Minions
- Skeletons
- SkeletonArmy
- Balloon
- Witch
- Barbarians
- Golem
- Valkyrie
- Bomber
- Musketeer
- BabyDragon
- Prince
- Wizard
- SpearGoblins
- GiantSkeleton
- HogRider
- MinionHorde
- RoyalGiant
- Princess
- ThreeMusketeers (Not the newest version though)
- BlowdartGoblin (Before nerf)
- AngryBarbarians (English name: Elite Barbarians)
- Bats
- DartBarrell (English name: Flying Machine)
- RoyalHogs
- Cannon
- Xbow
- IceWizard
- SkeletonWarriors
- DarkPrince
- LavaHound
- IceSpirits
- FireSpirits
- Miner
- Sparky
- Bowler
- Rage
- RageBarbarian (English name: Lumberjack)
- BattleRam

## Where is the code?

The folder structure of this repo is very complicated. I'm sorry for the inconvenience!

The real code files are in `src/clasher_new`. 
- `__init__.py` is an empty placeholder
- `gamedata.json` contains very necessary data extracted from the game, like hitpoints, damage, etc.
- `card_utils.py` reads `gamedata.json` and provides easy ways to access character attributes
- `arena.py` defines `TileGrid` which contains information on where each sides' King tower and princess towers are located
- `player.py` is a short file that stores a player's information in game, like current elixir.
- `battle.py` contains all the game logic, defining behavior for troops, buildings, projectiles and other mechanics.

## I need help

This project is far from finishing. I already poured more than 25 hours into this project and many more 
still lies ahead. If you want to contribute, please submit issues or pull requests. 

You are more than welcome to contact me via my email: `2243272839@qq.com`