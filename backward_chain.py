from __future__ import annotations
import os

# os.environ.setdefault("DEBUG", 'false')

import gym
from core import *
from minigrid_adapters import *
import pygame
import random
from minigrid.wrappers import ActionBonus
from minigrid.core.constants import *
import utils
import rl_adapters
import argparse
import numpy
import utils
from torch_ac.utils.penv import ParallelEnv
import torch
from utils import device
import matplotlib.pyplot as plt
import json
import yaml

MAX_DEPTH = 10


def deep_condition(condition: Node | None, depth: int = 1) -> Node | None:
    if condition is None:
        return None

    if depth >= 10:
        return condition

    if isinstance(condition, CompositeNode):
        for i, child in enumerate(condition.children):
            condition.replace_child(i, deep_condition(child, depth=depth + 1))
    else:
        con = deep_condition(condition.condition(), depth=depth + 1)
        if con is not None:
            return Sequence(con, condition)

    return condition


def deep_effect(effect: Node | None, depth: int = 1) -> Node | None:
    if effect is None:
        return None

    if depth >= 10:
        return effect

    if isinstance(effect, CompositeNode):
        for i, child in enumerate(effect.children):
            effect.replace_child(i, deep_effect(child, depth=depth + 1))
    else:
        eff = deep_effect(effect.effect(), depth=depth + 1)
        if eff is not None:
            return Selector(eff, effect)

    return effect


def condition_tag(condition: Node) -> str:
    # 返回条件节点的标签（去除Or的影响）
    if isinstance(condition, CompositeNode) and len(condition.children) == 1:
        return condition.children[0].to_xml(sorted_children=True)
    return condition.to_xml(sorted_children=True)


def is_condition_equal(x: Node, y: Node) -> bool:
    # 返回条件节点的标签（去除Or的影响）
    return condition_tag(x) == condition_tag(y)


def match_action_nodes_by_goal(goal: Node) -> [Node]:
    if not isinstance(goal, ConditionNode):
        return None

    action_nodes = []
    for node_cls in REGISTER_NODES:
        try:
            node = node_cls()
        except Exception as e:
            print(f'{node_cls.__name__}: {e}')
            raise e
        node_effect = deep_effect(node.effect())
        if node_effect is None:
            continue

        if is_effect_contains_goal(node_effect, goal):
            action_nodes.append(node)

    return action_nodes


def is_effect_contains_goal(effect: Node, goal: Node) -> bool:
    # 判断effect是否包含了goal
    # goal为true的时候effect有可能为true
    effect_xml_node = effect.to_xml_node()
    goal_xml_node = goal.to_xml_node()

    goal_state = utils.expression_build_state(goal)
    effect_state = utils.expression_build_state(effect)

    for key in goal_state:
        if key in effect_state:
            del effect_state[key]

    for item_goal_state in utils.expression_generate_all_possible_states(goal_state):
        if utils.expression_evaluate(goal_xml_node, item_goal_state):
            for item_effect_state in utils.expression_generate_all_possible_states(effect_state):
                if utils.expression_evaluate(effect_xml_node, {
                    **item_goal_state, **item_effect_state
                }):
                    print('is_condition_contains_goal True', effect, goal)
                    return True
    print('is_condition_contains_goal False', effect, goal)
    return False


def generate(goal: Node, depth: int = 1, memory: set = None) -> Node:
    print('generating node', goal.tag(), depth)
    if depth + 1 > MAX_DEPTH:
        return goal

    if memory is None:
        memory = set()

    if isinstance(goal, ConditionNode):
        action_nodes = match_action_nodes_by_goal(goal)
        if len(action_nodes) == 0:
            return goal

        if len(action_nodes) == 1:
            action_node = action_nodes[0]
            action_node_condition = deep_condition(action_node.condition())
            if action_node_condition is None and depth + 1 < MAX_DEPTH:
                action_node = Sequence(action_node_condition, action_node)
                action_node = generate(action_node, depth=depth + 2, memory={
                    *memory, goal.tag()
                })
            return Selector(goal(), action_node)
        else:
            results = []
            for action_node in action_nodes:
                action_node_condition = deep_condition(action_node.condition())
                if action_node_condition is not None and depth + 1 < MAX_DEPTH:
                    action_node = Sequence(action_node_condition, action_node)
                    action_node = generate(action_node, depth=depth + 2, memory={
                        *memory, goal.tag()
                    })
                results.append(action_node)
            return Selector(goal(), *results)
    elif isinstance(goal, Sequence):
        for i, item in enumerate(goal.children):
            goal.replace_child(i, generate(item, depth=depth + 1))

    return goal


if __name__ == '__main__':
    # print(len(list(utils.expression_generate_all_possible_states(state))))
    # print(evaluate_expression(ExploreUnseen().effect().to_xml(), state))

    # 运行backward_chain
    goal = Sequence(IsReachGoal())
    goal = generate(goal=goal)
    with open('goal.xml', 'w') as f:
        f.write(goal.to_xml())

    # 运行仿真
    node = Node.from_xml_file('goal.xml')
    render_mode = 'human'
    env = MemoryImageObsWrapper(gym.make('MiniGrid-DoorKey-5x5-v0', render_mode=render_mode))
    simulation = MiniGridSimulation(env=env)
    simulation.train = False
    simulation.workspace = '.'
    simulation.add_child(node)
    simulation.reset(seed=0)

    pygame.init()
    pygame.display.set_caption(f'MiniGrid-DoorKey-16x16-v0 [{simulation.seed}]')
    simulation.train = False
    for episode in range(10):
        simulation.reset()
        for i in range(10000):
            simulation.tick()
            if simulation.done:
                break
    simulation.close()
