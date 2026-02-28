import numpy as np

gamma = 0.9
states = 5  # stock levels
V = np.zeros(states)
cost = -2
profit = 5

for _ in range(100):
    for s in range(states):
        order = max(0, 4-s)
        V[s] = profit - cost*order + gamma*V[min(4,s+1)]

print("Optimal Stock Values:", V)
