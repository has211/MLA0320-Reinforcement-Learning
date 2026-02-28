import numpy as np

gamma = 0.9
states = 5
V = np.zeros(states)

for _ in range(100):
    for s in range(states):
        V[s] = max(s + gamma*V[(s+1)%states],
                   s + gamma*V[(s-1)%states])

print("Optimal Values:", V)
