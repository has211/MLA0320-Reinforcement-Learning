import numpy as np

theta = 0.5
alpha = 0.01

for _ in range(1000):
    action = np.random.rand()
    reward = action * np.random.randn()
    theta += alpha * reward

print("Optimized Parameter:", theta)
