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

## 项目结构

如果你也对训练皇室战争的AI模型非常感兴趣，那么我强烈建议你去认真阅读我这个仓库的代码。模拟器的代码总量应该两千行左右，而模型训练的代码则有约500行代码（我没统计过，纯个人感觉）。比如说，你想知道寻路机制是怎么实现的？我的模拟器是怎么处理索敌的？我的AI模型架构是什么，RL环境的观测空间和动作空间是什么？

我这个项目绝大部分的代码都是自己一行一行敲出来的，如果你想针对某个bug提出修改方案，或者增加模拟器的功能，提出的PR中不能有明显的AI痕迹。 

模拟器的实现：

```plaintext
短而且逻辑简单，定义一些常用的类供模拟器使用
arena.py
core.py
player.py
card_utils.py

核心的战斗引擎、寻路机制和卡牌逻辑实现
battle.py
card_mechanics.py
new_visualization.py
pathfinding.py
pathfinding_heap.py

联机对战
server.py
client_side/client.py
```

RL环境和训练代码：
```plaintext
environment.py
train.py
```

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

## 帮个忙吧

不知不觉从项目开始开发到现在，已经过去半年了，我已经数不清有多少时间花在上面了。然而，这个项目的上限基本取决于模拟器的上限，我一人无力准确实现皇室122张卡牌的所有逻辑，如果你愿意帮我实现一两张牌，我将会非常高兴。

我的B站用户名是`jasonmoonw`，如果你想联系我，给我发私信就好了。我的discord用户名是jasoncoder_47308。

给我的项目点个star吧！
