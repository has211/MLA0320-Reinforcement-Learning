import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CleaningRobotEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.grid_size = 5
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(25)

        self.dirt = [(1,1), (2,3), (4,4)]
        self.obstacles = [(1,3), (3,2)]
        self.state = (0,0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = (0,0)
        return self.state, {}

    def step(self, action):
        x, y = self.state

        if action == 0 and x > 0: x -= 1
        if action == 1 and x < 4: x += 1
        if action == 2 and y > 0: y -= 1
        if action == 3 and y < 4: y += 1

        reward = 0
        if (x,y) in self.dirt:
            reward = 1
        elif (x,y) in self.obstacles:
            reward = -1

        self.state = (x,y)
        terminated = False
        truncated = False
        return self.state, reward, terminated, truncated, {}

env = CleaningRobotEnv()
state, info = env.reset()

for _ in range(10):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    print("State:", state, "Reward:", reward)
