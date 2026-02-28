import numpy as np
import matplotlib.pyplot as plt

gamma = 0.9
states = 5
V = np.zeros(states)

for _ in range(100):
    for s in range(states):
        V[s] = 1 + gamma * V[(s+1)%states]

plt.plot(V)
plt.title("Value Function")
plt.show()
