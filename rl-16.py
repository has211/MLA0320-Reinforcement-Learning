import numpy as np
import random

content = 5
Q = np.zeros(content)
N = np.zeros(content)
epsilon = 0.1

for _ in range(1000):
    if random.random() < epsilon:
        c = random.randint(0,content-1)
    else:
        c = np.argmax(Q)

    click = random.choice([0,1])
    N[c]+=1
    Q[c]+= (click-Q[c])/N[c]

print("Best Recommended Content:", np.argmax(Q))
