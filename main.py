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

# Parse arguments

parser = argparse.ArgumentParser()

parser.add_argument("--w", type=str, required=True,
                    help="workspace directory (REQUIRED)")

args = parser.parse_args()
# Set seed for all randomness sources
WORKSPACE = args.w
print(f"Device: {device}\n")
with open(os.path.join(WORKSPACE, 'config.yaml'), 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

SEED = config['seed']
utils.seed(SEED)
TRAIN = config['train']
EVALUATE = config['evaluate']


def on_tick(node):
    print('on_tick')
    output_dir = os.path.join('output', 'MiniGrid-DoorKey-Ticks', str(SEED))
    os.makedirs(output_dir, exist_ok=True)
    xml_text = node.to_xml(effect_score=True, status=True, seed=True)

    with open(os.path.join(output_dir, f'log.xml'), 'w', encoding='utf-8') as f:
        f.write(xml_text)

    with open(os.path.join(output_dir, f'tick-{node.tick_count}.xml'), 'w', encoding='utf-8') as f:
        f.write(xml_text)


# def demo_door_key():
#     node = Node.from_xml_file('workspace/DoorKey/MiniGrid-DoorKey.xml')
#     env = gym.make("MiniGrid-DoorKey-16x16-v0", render_mode='human')
#     env = MemoryImageObsWrapper(env)
#     simulation = MiniGridSimulation(env=env)
#     simulation.add_child(node)
#     simulation.gif = args.gif
#     simulation.reset(seed=SEED)
#
#     pygame.init()
#     pygame.display.set_caption(f'MiniGrid-DoorKey-16x16-v0 [{simulation.seed}]')
#
#     for i in range(10000):
#         simulation.tick()
#         simulation.render()
#         print('i: ', i)
#         if simulation.done:
#             break
#     simulation.close()

# def demo_manual_control():
#     node = Node.from_xml_file('workspace/ManualControl/MiniGrid-ManualControl.xml')
#     env = gym.make("MiniGrid-DoorKey-16x16-v0", render_mode='human')
#     env = ActionBonus(env)
#     simulation = MiniGridSimulation(env=env)
#     simulation.add_child(node)
#     simulation.reset(seed=SEED)
#
#     pygame.init()
#     pygame.display.set_caption(f'MiniGrid-DoorKey-16x16-v0 [{simulation.seed}]')
#
#     for i in range(10000):
#         simulation.tick()
#         print('reward', simulation.step_results[-1].reward)
#         simulation.render()
#         print('i: ', i)
#         if simulation.done:
#             break
#         time.sleep(0.1)
#     simulation.close()

node = Node.from_xml_file(WORKSPACE)
render_mode = 'human'
if config['train'] or config['evaluate']:
    render_mode = 'rgb_array'

env = MemoryImageObsWrapper(gym.make(config['env'], render_mode=render_mode))
simulation = MiniGridSimulation(env=env)
simulation.train = config['train']
simulation.workspace = WORKSPACE
simulation.add_child(node)
simulation.reset(seed=SEED)


def evaluate():
    # MiniGrid-Empty-16x16-v0
    simulation.train = False
    start_time = time.time()

    num_frames_per_episode = []
    return_per_episode = []
    terminated_per_episode = []
    episodes = list(range(1, config['episodes'] + 1))

    for episode in episodes:

        total_reward = 0
        simulation.reset()
        frame = 0
        for i in range(10000):
            frame = i
            simulation.tick()
            last_rs = simulation.step_results[-1]
            if simulation.done:
                break

        terminated_per_episode.append(last_rs.terminated)
        num_frames_per_episode.append(frame)

        for item in simulation.step_results:
            if item.reward is not None:
                total_reward += item.reward

        final_reward = simulation.step_results[-1].reward
        return_per_episode.append(total_reward)
        print('episode', episode, 'terminated', last_rs.terminated, 'truncated', last_rs.truncated, 'frame', frame,
              'total_reward', total_reward, 'final_reward', final_reward)

    simulation.close()

    end_time = time.time()

    # Print logs

    num_frames = sum(num_frames_per_episode)
    fps = num_frames / (end_time - start_time)
    duration = int(end_time - start_time)

    terminated_count = 0
    for terminated in terminated_per_episode:
        if terminated:
            terminated_count += 1

    log = {
        'num_frames'            : num_frames,
        'return_per_episode'    : utils.synthesize(return_per_episode),
        'num_frames_per_episode': utils.synthesize(num_frames_per_episode),
        'duration'              : duration,
        'fps'                   : fps,
        'terminated_count'      : terminated_count
    }

    with open(os.path.join(WORKSPACE, 'log.json'), 'w') as f:
        json.dump(log, f, indent=4, cls=utils.CustomJSONEncoder)

    utils.plot_and_save(
            x=episodes,
            y=return_per_episode,
            title='Return per Episode',
            xlabel='Episode',
            ylabel='Return',
            savefig=os.path.join(WORKSPACE, 'return_per_episode.png'))

    utils.plot_and_save(
            x=episodes,
            y=num_frames_per_episode,
            title='Frames per Episode',
            xlabel='Episode',
            ylabel='Frame',
            savefig=os.path.join(WORKSPACE, 'num_frames_per_episode.png'))


def train():
    # MiniGrid-Empty-16x16-v0
    simulation.train = True
    for episode in range(config['episodes']):
        simulation.reset()
        for i in range(config['episode_max_frame']):
            simulation.tick()
            if simulation.done:
                break
        if config['gif']:
            from array2gif import write_gif
            write_gif(numpy.array(simulation.gif_frames), config['gif'] + ".gif", fps=1 / config['pause'])

    simulation.close()


def visualize():
    pygame.init()
    pygame.display.set_caption(f'MiniGrid-DoorKey-16x16-v0 [{simulation.seed}]')
    simulation.train = False
    for episode in range(config['episodes']):
        simulation.reset()
        for i in range(10000):
            simulation.tick()
            if simulation.done:
                break
        if config['gif']:
            from array2gif import write_gif
            write_gif(numpy.array(simulation.gif_frames), config['gif'] + ".gif", fps=1 / config['pause'])

    simulation.close()


if __name__ == '__main__':
    if config['evaluate']:
        evaluate()
    elif config['train']:
        train()
    else:
        visualize()
