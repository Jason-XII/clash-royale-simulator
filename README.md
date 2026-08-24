# Clash Royale Simulator 

项目完整介绍视频：https://www.bilibili.com/video/BV1n3uZ6WE5P/

众所周知，想让一个AI学会怎样玩好一款游戏，必须让它收集大量的经验。对皇室战争这款游戏来说，经验收集是人机开发和训练的一个巨大瓶颈，哪怕是掌握内部战斗引擎接口的supercell员工，也没有办法加速引擎来更快的收集经验（Learning to Play Imperfect-Information Games by Imitating an Oracle Planner, arXiv:2012.12186）。

而我的仓库则依赖于自建的完整模拟器逻辑，来弥补目前所有AI训练的经验收集瓶颈。这个模拟器读取准确的游戏数值，复现了原始游戏引擎的A*搜索寻路逻辑，卡牌交互基本准确。这数千行代码全由我一人编写，无AI Agent辅助。经过不断的优化，目前的模拟器性能可以做到1.1秒左右跑完180秒的对局，在我的M4芯片上实现大概150倍的加速。在一个一般般的CPU上，大概也能做到70~90倍相对于真实时间的加速。

在模拟器基础之上，为了方便所有人使用我的模拟器进行AI的研究，我还搭建了一个强化学习环境，可以通过Stable-Baselines3即插即用进行训练，目前使用基于CNN的网络架构，模型能够稳定的学习和进步。

## 效果展示

![demo](./demo2.gif)

上图为模拟器界面与游戏实际效果的对比。我记录了真实游戏里的下牌时间，然后输入到模拟器中进行模拟，在前三十秒的对局中，卡牌交互是完全准确的。

## 安装

在终端中运行下面的命令：
```bash
git clone https://github.com/Jason-XII/clash-royale-simulator.git
cd clash-royale-simulator
pip install pygame fastcore numpy stable-baselines3 tensorboard --user --no-cache-dir
```

## 局域网联机

本模拟器支持局域网联机功能，也就是说，你可以和你的朋友在局域网内联机进行对战。我开发这个功能只是为了快速测试卡牌的效果，并不是实现了一个皇室战争服务器。联机步骤如下：

1. 找到本机在局域网的IP地址。在Windows系统上运行`ipconfig`，在MacOS系统上运行`ifconfig | grep inet`即可得到本机的IP地址，通常以192.168开头。
2. 在`src/clasher_new/server.py`的最后找到存放ip地址的位置，把它替换为你自己的IP地址。然后运行。
3. 在两台电脑上同时运行`src/clasher_new/client_side/client.py`，选择卡组后，输入刚才的IP地址即可连接。
4. 两个客户端都连接后，游戏会自动开始。


## 模拟器特性

目前我实现了47张卡牌，因为时间精力有限，暂时没有实现觉醒、精英和英雄卡的打算。模拟器有着和原游戏一致的寻路算法，大部分角色有和游戏相同的数值。

下面是我实现的所有卡牌名称：

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
- Fireball
- Arrows

## Read the code

If you are interested in this project, you can consider reading my code and figure out
how the simulator work for yourself. I tried my best to write clear code. Here's a brief
explanation of each python file in the repo, listed in suggested reading order:

The real code files are in `src/clasher_new`. 
- `__init__.py` is an empty placeholder
- `gamedata.json` contains very necessary data extracted from the game, like hitpoints, damage, etc.
- `cards_stats_xxx.json` files are also game data files downloaded from *royaleapi.com*.
- `card_utils.py` reads `gamedata.json` and provides easy ways to access character attributes
- `arena.py` defines `TileGrid` which contains information on where each sides' King tower and princess towers are located
- `player.py` is a short file that stores a player's information in game, like current elixir.
- `battle.py` contains all the game logic, defining behavior for troops, buildings, projectiles and other mechanics.
- `core.py` and `card_mechanics.py` provides an interface for special card logic. Makes the system more flexible.
- `server.py` and `client.py` gives a simple pygame interface that allows two players to connect through local network and play in realtime.

## To-do list

I trained the model for 8M steps and the win rate stops improving at around 70-80%. This is probably
caused by the model trying to independently predict the card, position x and position y.
Humans make decisions by choosing all three simultaneously, we first determine (vaguely)
what we should do, and then do the actual placement. So I think the model can perform better
if I let it produce the x and y position at once, or we use a different structure entirely
by using the selection attention framework introduced in another paper. I might have to 
learn more about that paper. 

Against a random agent, the PPO model should train relatively well and be consistently winning
after 1-2M steps. So something's probably wrong with the model architecture.

I also thought of a way to determine the level of game playing agents: tier testing.
Imagine we have a pool of agents and we let them fight for enough rounds. Good enough agents will 
consistently win and those with winrates over 70% can make it to the next tier. The logic 
might be a little flawed here but points to the right direction.

(might introduce elo score?)

Lots of new benchmarks can be added besides the winrate against a random agent. For example,
we can test the agent's use of spells by placing swarm units at the bridge. By placing a
mini pekka behind a giant we test the agent's ability of defending. 

The environment can also be refined. The agent needs to know the phase of the game that 
it's currently in, because decks have different strategies in single, double and triple elixir.
When it's near the end of the game and overtime, the agent may need to cycle spells to win.

The biggest flaw in the simulator is that the king tower activation seems a bit off. 
The king tower seems to have a shorter range than the musketeer, which is very strange.
But this can be easily fixed.


## I need help

This project is far from finishing. I already poured more than 100 hours into this project and many more 
still lies ahead. If you want to contribute, please submit issues or pull requests. 

You are more than welcome to contact me via my email: `2243272839@qq.com` 

Or you can add my discord: jasoncoder_47308