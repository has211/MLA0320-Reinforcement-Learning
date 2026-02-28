import gymnasium as gym
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

env = gym.make("MountainCar-v0")

model = keras.Sequential([
    layers.Dense(24, activation='relu', input_shape=(2,)),
    layers.Dense(24, activation='relu'),
    layers.Dense(3, activation='linear')
])

model.compile(optimizer='adam', loss='mse')

state, _ = env.reset()

for _ in range(10):
    action = env.action_space.sample()
    state, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        state, _ = env.reset()

print("Environment ran successfully")
env.close()
