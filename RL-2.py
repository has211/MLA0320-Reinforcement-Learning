import numpy as np

gamma = 0.9
states = 5
V = np.zeros(states)
rewards = [0,2,5,-2,1]

for _ in range(100):
    for s in range(states):
        V[s] = rewards[s] + gamma * V[(s+1)%states]

print("Value Function:", V)
