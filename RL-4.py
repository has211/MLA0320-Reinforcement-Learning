import numpy as np

gamma = 0.9
states = 4
V = np.zeros(states)
policy = np.zeros(states)

for _ in range(50):
    # Policy Evaluation
    for s in range(states):
        V[s] = s + gamma * V[(s+1)%states]

    # Policy Improvement
    for s in range(states):
        policy[s] = np.argmax([V[(s+1)%states], V[(s-1)%states]])

print("Optimal Policy:", policy)
