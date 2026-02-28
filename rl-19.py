import numpy as np

gamma = 0.9
V = np.zeros(4)

policy = [1,1,1,0]  # 1=move forward

for _ in range(100):
    for s in range(4):
        if policy[s]==1:
            V[s] = 1 + gamma*V[(s+1)%4]
        else:
            V[s] = 0

print("Value Function under Policy:", V)
