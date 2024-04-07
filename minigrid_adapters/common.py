from __future__ import annotations

import utils
from utils import AStar
from minigrid.core.constants import *
from typing import Union, Iterable
from itertools import islice
import numpy as np

class _MiniGridAStar(AStar):

    def __init__(self, memory_obs: np.ndarray, goal: (int, int) = None):
        self.memory_obs = memory_obs
        self.goal = goal

    def heuristic_cost_estimate(self, current, goal) -> float:
        """
        计算启发式距离，使用曼哈顿距离
        :param current:
        :param goal:
        :return:
        """
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

    def distance_between(self, n1, n2) -> float:
        """
        计算两个节点之间的距离, 使用曼哈顿距离
        n2 is guaranteed to belong to the list returned by the call to neighbors(n1).
        :param n1:
        :param n2:
        :return:
        """
        distance = manhattan_distance(n1, n2)

        if n2 == self.goal:
            # 如果n2就是目标节点，那么可以移动过去
            return distance

        n1_object_idx, _, n1_state = self.memory_obs[n1[0], n1[1], :]
        n2_object_idx, _, n2_state = self.memory_obs[n2[0], n2[1], :]
        n1_object = IDX_TO_OBJECT[n1_object_idx]
        n2_object = IDX_TO_OBJECT[n2_object_idx]

        if n1_object in ['wall', 'ball', 'box', 'lava']:
            # 不能穿过墙壁、钥匙、球、箱子、岩浆
            return float('inf')

        if n2_object in ['wall', 'key', 'ball', 'box', 'lava', 'agent']:
            # 不能穿过墙壁、钥匙、球、箱子、岩浆、agent
            return float('inf')

        if n2_object == 'door' and n2_state != STATE_TO_IDX['open']:
            # 不能穿过关闭的门
            return float('inf')

        return distance

    def neighbors(self, node):
        """
        返回当前节点的邻居节点
        :param node:
        :return:
        """
        for dx, dy in DIR_TO_VEC:
            x2 = node[0] + dx
            y2 = node[1] + dy
            if x2 < 0 or x2 >= self.memory_obs.shape[0] or y2 < 0 or y2 >= self.memory_obs.shape[1]:
                continue
            yield x2, y2


# def is_match_obs(obs: (int, int, int), target: (int, int, int)) -> int:
#     """
#     将object转换为对应的idx
#     :param object:
#     :return:
#     """
#     target_object, target_color, target_state = target
#     obs_object, obs_color, obs_state = obs
#


# 找到离自己最近的物体
def find_nearest_object_pos(
        obj: str,
        memory_obs: np.ndarray,
        agent_pos: (int, int), color: str = '',
        near_range: (int, int) = (0, 1e6)) -> (int, int):
    """
    找到离自己最近的物体
    :param obj:
    :param memory_obs:
    :param agent_pos:
    :param color:
    :param near_range: [min_distance, max_distance]
    :return:
    """
    min_distance = 1e6
    door_pos = None
    for x in range(memory_obs.shape[0]):
        for y in range(memory_obs.shape[1]):
            distance = abs(x - agent_pos[0]) + abs(y - agent_pos[1])
            if distance < near_range[0] or distance > near_range[1]:
                continue
            object_idx, color_idx, state = memory_obs[x, y, :]
            if object_idx == OBJECT_TO_IDX[obj] and (color == '' or color_idx == COLOR_TO_IDX[color]):
                if distance < min_distance:
                    min_distance = distance
                    door_pos = (x, y)

    return door_pos


# 曼哈顿距离
def manhattan_distance(pos1: (int, int), pos2: (int, int)) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def astar_find_path(obs, start, target) -> Union[Iterable[(int, int)], None]:
    """
    使用A*算法寻找路径
    :param obs:
    :param start:
    :param target:
    :return:
    """
    astar = _MiniGridAStar(memory_obs=obs, goal=target)
    path = astar.astar(start, target)
    return path

def iter_take(iterable, n):
    """
    从迭代器中取出n个元素
    :param iterable: 迭代器
    :param n: 取出的元素个数
    :return:
    """
    return list(islice(iterable, n))
