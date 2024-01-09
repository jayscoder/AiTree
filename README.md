# The-Python-code-implements-aitree

a powerful Python library designed for crafting customizable Behavior Trees and integrating them seamlessly with Reinforcement Learning systems.


## minigrid

http://www.jidiai.cn/env_detail?envid=24

图像表示: Minigrid观察通常是一个二维图像，可以用一个矩阵来表示。这个矩阵的每个元素表示图像中的一个像素点。

智能体位置: 图像中通常包含一个标记了智能体位置的元素。这可以是一个特定颜色、形状或者其他可识别的标记，用于表示智能体在环境中的当前位置。

环境布局: 图像捕捉了环境中的布局，包括墙壁、门、宝藏等。每个物体都可能以图像中的不同形式呈现，以便智能体能够感知和理解环境。

颜色编码或符号: 不同类型的物体通常使用不同的颜色或符号进行编码。这有助于智能体区分和理解环境中的不同元素。

可见范围: 观察可能是智能体当前位置周围的部分环境，而不是整个环境的图像。这模拟了真实世界中智能体的有限感知能力。

MiniGrid是一款简约2D环境模拟器，用于强化学习和机器人技术的研究。其中，世界的结构是：

世界是NxM的网格
世界中的每个网格都包含零个或一个对象
每个对象都有一个颜色
每个对象都有一个关联的类型
智能体可以拾起并随身携带一个物体
要打开上锁的门，智能体必须携带与门的颜色匹配的钥匙


(OBJECT_IDX, COLOR_IDX, STATE)
OBJECT_IDX: 0: nothing, 1: wall, 2: door, 3: key, 4: ball, 5: box, 6: goal, 7: lava 8: agent
COLOR_IDX: 0: red, 1: green, 2: blue, 3: purple, 4: yellow, 5: grey
STATE: 0: open, 1: closed, 2: locked

左上角的坐标是(1,1)，右下角的坐标是(M, N)
如果物体被智能体拾起，那么它的坐标就是智能体的坐标

智能体的可视空间是一个正方形，它的边长是obs['image'].shape
