from __future__ import annotations

import gymnasium

from .config import *
from minigrid.core.constants import *
from minigrid.envs.empty import MiniGridEnv
from .common import astar_find_path, manhattan_distance, iter_take
from core.nodes import Node, NODE_STATUS, SUCCESS, FAILURE, RUNNING, register, Simulation
from typing import Union, List
import gymnasium as gym
import random

from minigrid.wrappers import ObservationWrapper, ObsType, ImgObsWrapper, Any, DirectionObsWrapper


# @register(
#         props=[
#             {
#                 'name'   : 'render_mode',
#                 'type'   : 'str',
#                 'default': 'human',
#                 'desc'   : 'The render mode of the environment',
#                 'options': ['human', 'rgb_array']
#             },
#         ]
# )
class MiniGridSimulation(Simulation):
    @property
    def width(self) -> int:
        return self.env.unwrapped.width

    @property
    def height(self) -> int:
        return self.env.unwrapped.height

    @property
    def front_pos(self) -> (int, int):
        pos = self.env.front_pos
        return pos[0] - 1, pos[1] - 1

    @property
    def dir_vec(self) -> (int, int):
        vec = DIR_TO_VEC[self.agent_dir]
        return vec[0], vec[1]

    @property
    def agent_pos(self) -> (int, int):
        pos = self.env.unwrapped.agent_pos
        return pos[0] - 1, pos[1] - 1

    @property
    def agent_dir(self) -> int:
        return self.env.unwrapped.agent_dir

    @property
    def memory_obs(self) -> np.ndarray:
        return self.step_results[-1].obs['memory_image']

    def get_obs_item(self, pos: (int, int)) -> ObsItem | None:
        """
        获取指定位置的物体
        :param pos: 位置
        :return:
        """
        if pos[0] < 0 or pos[0] >= self.width or pos[1] < 0 or pos[1] >= self.height:
            return None

        object_idx, color_idx, state_idx = self.memory_obs[pos[0], pos[1], :]
        return ObsItem(
                obj=IDX_TO_OBJECT[object_idx],
                color=IDX_TO_COLOR[color_idx],
                state=IDX_TO_STATE[state_idx],
                pos=pos)

    def find_can_reach_obs(self, obj: str, color: str = '') -> ObsItem | None:
        """
        找到能够到达的物体
        :param obj:
        :param color:
        :return: (object_idx, color_idx, state)
        """
        memory_obs = self.memory_obs
        for x in range(memory_obs.shape[0]):
            for y in range(memory_obs.shape[1]):
                object_idx, color_idx, state = memory_obs[x, y, :]
                if object_idx == OBJECT_TO_IDX[obj] and (color == '' or color_idx == COLOR_TO_IDX[color]):
                    if self.can_move_to(target=(x, y)):
                        return self.get_obs_item((x, y))
        return None

    def find_nearest_obs(
            self,
            obj: str,
            color: str = '',
            near_range: (int, int) = (0, 1e6)) -> ObsItem | None:
        """
        找到离自己最近的物体
        :param obj:
        :param color:
        :return: (object_idx, color_idx, state)
        """
        memory_obs = self.memory_obs
        agent_pos = self.agent_pos

        min_distance = 1e6
        min_pos = None

        for x in range(memory_obs.shape[0]):
            for y in range(memory_obs.shape[1]):
                distance = abs(x - agent_pos[0]) + abs(y - agent_pos[1])
                if distance < near_range[0] or distance > near_range[1]:
                    continue
                object_idx, color_idx, state = memory_obs[x, y, :]
                if object_idx == OBJECT_TO_IDX[obj] and (color == '' or color_idx == COLOR_TO_IDX[color]):
                    if distance < min_distance:
                        min_distance = distance
                        min_pos = (x, y)
        if min_pos is None:
            return None
        return self.get_obs_item(min_pos)

    def move_forward(self, target: int | (int, int) | None = None) -> NODE_STATUS:
        """
        向前移动step步, 如果direction不为空，则按照direction的方向移动，否则按照agent_dir的方向移动
        :param target: 方向或目标位置
        :return:
        """
        agent_pos = self.agent_pos
        agent_dir = self.agent_dir
        if target is None:
            direction = agent_dir
        elif isinstance(target, int):
            direction = target
        else:
            direction = self.get_target_direction(target)
            if direction is None:
                return SUCCESS(msg='已经在目标位置, 不需要移动')

        self.turn_to(target=direction)

        # 判断是否还能继续前进
        if direction == Directions.right:
            if agent_pos[0] == self.width - 1:
                return FAILURE(msg='不能继续朝着右侧移动')
        elif direction == Directions.down:
            if agent_pos[1] == self.height - 1:
                return FAILURE(msg='不能继续朝着下方移动')
        elif direction == Directions.left:
            if agent_pos[0] == 0:
                return FAILURE(msg='不能继续朝着左侧移动')
        elif direction == Directions.up:
            if agent_pos[1] == 0:
                return FAILURE(msg='不能继续朝着上方移动')

        self.step(Actions.forward)

        return SUCCESS

    def turn_to(self, target: int | (int, int) | None) -> NODE_STATUS:
        """
        转向到目标或目标方向，如果target为None，则不转向
        :param target: 目标方向或目标未知
        :return:
        """
        if target is None:
            return SUCCESS(msg='已经在目标位置，不需要转向')

        if isinstance(target, int):
            direction = target
        elif isinstance(target, ObsItem):
            direction = self.get_target_direction(target.pos)
        else:
            direction = self.get_target_direction(target)

        if direction is None:
            return SUCCESS(msg='已经在目标位置，不需要转向')

        agent_dir = self.agent_dir
        if direction == agent_dir:
            return SUCCESS(msg='已经在目标方向')
        elif direction == (agent_dir + 1) % 4:
            # 右转
            self.step(Actions.right)
        elif direction == (agent_dir + 2) % 4:
            self.step(Actions.right)
            self.step(Actions.right)
        elif direction == (agent_dir + 3) % 4:
            self.step(Actions.left)

        return SUCCESS

    def get_target_direction(self, target: (int, int)) -> int | None:
        """
        获取目标位置相对于自己的方向，如果目标位置就是自己的位置，则返回None
        :param target: 目标位置
        :return:
        """
        agent_pos = self.agent_pos
        if target[0] > agent_pos[0]:
            direction = Directions.right
        elif target[0] < agent_pos[0]:
            direction = Directions.left
        elif target[1] > agent_pos[1]:
            direction = Directions.down
        elif target[1] < agent_pos[1]:
            direction = Directions.up
        else:
            return None
        return direction

    def can_move_to(self, target: (int, int)) -> bool:
        path = astar_find_path(obs=self.memory_obs, start=self.agent_pos, target=target)
        return path is not None

    def move_to(self, target: (int, int), step: int = 1, nearby: int = 0):
        """
        移动到目标位置
        :param target: 目标位置
        :param step: 移动步数
        :param nearby: 是否只移动到目标位置附近
        :return:
        """
        if manhattan_distance(self.agent_pos, target) <= nearby:
            return SUCCESS(msg='已经在目标位置')

        start_agent_pos = self.agent_pos
        path = astar_find_path(obs=self.memory_obs, start=self.agent_pos, target=target)

        if path is None:
            return FAILURE(msg='路径不存在 path=None')
        path = iter_take(path, step + 1)

        for i in range(1, len(path)):
            self.move_forward(target=path[i])
            if manhattan_distance(self.agent_pos, target) <= nearby:
                break

        if manhattan_distance(self.agent_pos, target) <= nearby:
            return SUCCESS

        if start_agent_pos == self.agent_pos:
            return FAILURE(msg=f'无法移动到目标位置 {target}')

        distance = manhattan_distance(self.agent_pos, target)
        max_size = self.width + self.height
        score = 1 - (distance / max_size) ** 0.8
        # 0-1
        score = min(1, max(0, score))
        return RUNNING(score=score, msg=f'正在移动到目标位置 {target}')


# plot score = 1 - (distance / max_size) ** 0.5
# x: distance
# y: score
# max_size: 16 + 16 = 32


class MemoryImageObsWrapper(ObservationWrapper):

    def __init__(self, env):
        """A wrapper that makes image the only observation.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.env_width = self.env.unwrapped.width
        self.env_height = self.env.unwrapped.height
        self._memory_obs = np.zeros((self.width, self.height, 3), dtype=np.uint8)

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ):
        # super
        obs, info = self.env.reset(seed=seed, options=options)
        self._memory_obs = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        return self.observation(obs), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(observation), reward, terminated, truncated, info

    def observation(self, obs):
        obs_image = obs['image']
        from minigrid.envs import EmptyEnv
        env: EmptyEnv = self.env

        for i in range(self.env_width):
            for j in range(self.env_height):
                pos = env.relative_coords(i + 1, j + 1)  #
                # 注意：这里的坐标是从1开始的，而不是从0开始的，所以要减1
                if pos is None:
                    continue
                obs_item = obs_image[pos[0], pos[1], :]
                if obs_item[0] == OBJECT_TO_IDX['unseen']:
                    # 墙壁会挡住视线，所以unseen不需要更新
                    continue
                self._memory_obs[i, j, :] = obs_item

        obs['memory_image'] = self._memory_obs
        return obs
