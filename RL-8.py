import gymnasium as gym

env = gym.make("FrozenLake-v1")
state, info = env.reset()

for _ in range(10):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    print("State:", state, "Reward:", reward)
    if terminated or truncated:
        state, info = env.reset()

env.close()
