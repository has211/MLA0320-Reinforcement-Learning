import numpy as np
import random

arms = 4
true = [0.1,0.3,0.5,0.7]

# Epsilon Greedy
eps = 0.1
Q = np.zeros(arms)
N = np.zeros(arms)

for _ in range(1000):
    if random.random()<eps:
        a=random.randint(0,arms-1)
    else:
        a=np.argmax(Q)
    reward=1 if random.random()<true[a] else 0
    N[a]+=1
    Q[a]+=(reward-Q[a])/N[a]

print("Best Campaign:", np.argmax(Q))
