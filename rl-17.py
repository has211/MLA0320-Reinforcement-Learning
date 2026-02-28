import numpy as np
import random

content = 5
Q = np.zeros(content)
N = np.ones(content)

for t in range(1,1000):
    ucb = Q + np.sqrt(np.log(t)/N)
    c = np.argmax(ucb)
    reward = random.choice([0,1])
    N[c]+=1
    Q[c]+= (reward-Q[c])/N[c]

print("Best Content by UCB:", np.argmax(Q))
