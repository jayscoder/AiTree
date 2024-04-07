from __future__ import annotations

"""
该文件定义了一些minigrid环境中的节点，用于构建Minigrid行为树
"""
from minigrid.core.actions import Actions
from minigrid.core.constants import *
from minigrid.envs.empty import MiniGridEnv
from enum import IntEnum

IDX_TO_STATE = dict(zip(STATE_TO_IDX.values(), STATE_TO_IDX.keys()))


class Objects(IntEnum):
    # "unseen": 0,
    # "empty": 1,
    # "wall": 2,
    # "floor": 3,
    # "door": 4,
    # "key": 5,
    # "ball": 6,
    # "box": 7,
    # "goal": 8,
    # "lava": 9,
    # "agent": 10,
    unseen = 0
    empty = 1
    wall = 2
    floor = 3
    door = 4
    key = 5
    ball = 6
    box = 7
    goal = 8
    lava = 9
    agent = 10

    def __eq__(self, other):
        if isinstance(other, Objects):
            return self.value == other.value
        elif isinstance(other, str):
            return self.name == other
        elif isinstance(other, int):
            return self.value == other
        else:
            return self.value == other

    def __str__(self):
        return f'{self.name}({self.value})'


OBJECTS_OPTIONS = ['empty', 'wall', 'floor', 'door', 'key', 'ball', 'box', 'goal', 'lava', 'agent']


class Directions(IntEnum):
    right = 0
    down = 1
    left = 2
    up = 3

    def __eq__(self, other):
        if isinstance(other, Directions):
            return self.value == other.value
        elif isinstance(other, str):
            return self.name == other
        elif isinstance(other, int):
            return self.value == other
        else:
            return self.value == other


DIRECTIONS_OPTIONS = ['right', 'down', 'left', 'up']


class Colors(IntEnum):
    red = 0
    green = 1
    blue = 2
    purple = 3
    yellow = 4
    grey = 5

    def __eq__(self, other):
        if isinstance(other, Colors):
            return self.value == other.value
        elif isinstance(other, str):
            return self.name == other
        elif isinstance(other, int):
            return self.value == other
        else:
            return self.value == other


COLOR_OPTIONS = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']


class States(IntEnum):
    open = 0
    closed = 1
    locked = 2


STATE_OPTIONS = ['open', 'closed', 'locked']


class ObsItem:
    def __init__(self, obj: str, color: str, state: str, pos: (int, int) = None):
        self.obj: Objects = Objects[obj]
        self.color: Colors = Colors[color]
        self.state: States = States[state]
        self.pos = pos

    def __str__(self):
        if self.pos is not None:
            return f'{self.obj.name}[{self.color.name}, {self.state.name}] ({self.pos[0]}, {self.pos[1]})'
        return f'{self.obj.name}[{self.color.name}, {self.state.name}]'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.obj == other.obj and self.color == other.color and self.state == other.state
