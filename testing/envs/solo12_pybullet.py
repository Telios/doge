import numpy as np

import gymnasium as gym
from gymnasium import utils
from gymnasium.spaces import Box, Dict
import torch, time, math, sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from external.quadruped_rl.PyBulletSimulator import PyBulletSimulator
from testing import doge_utils
from pybullet_utils import bullet_client

class Solo12Env(gym.Env):
    def __init__(self, healthy_reward=100.0):
        self.healthy_reward = healthy_reward
        self.action_space = Box(low=-1.0, high=1.0, shape=(12,))
        self.observation_space = Dict({
            "observation": Box(low=-np.inf, high=np.inf, shape=(12,)),
            "achieved_goal": Box(low=-np.inf, high=np.inf, shape=(12,)),
            "desired_goal": Box(low=-np.inf, high=np.inf, shape=(12,))
        })

    def step(self, action):
        pass

    def render(self):
        pass

    def close(self):
        pass

    def reset_model(self):
        pass

    def reset(self):
        pass

if __name__ == "__main__":
    env = Solo12Env()
    env.reset()
    for _ in range(1000):
        env.step(env.action_space.sample())
        
    env.close()