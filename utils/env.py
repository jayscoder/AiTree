import gymnasium as gym
import random

def make_env(env_key, seed=None, render_mode=None):
    env = gym.make(env_key, render_mode=render_mode)
    env.reset(seed=seed)
    return env

def random_seed():
    return random.randint(0, 10000000)
