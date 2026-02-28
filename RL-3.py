import numpy as np
import random

arms = 3
steps = 1000
true_values = [1,2,3]

# Epsilon Greedy
epsilon = 0.1
Q = np.zeros(arms)
N = np.zeros(arms)

for t in range(steps):
    if random.random() < epsilon:
        a = random.randint(0, arms-1)
    else:
        a = np.argmax(Q)

    reward = np.random.randn() + true_values[a]
    N[a] += 1
    Q[a] += (reward - Q[a]) / N[a]

print("Epsilon Greedy:", Q)

# UCB
Q = np.zeros(arms)
N = np.ones(arms)

for t in range(1, steps):
    a = np.argmax(Q + np.sqrt(np.log(t)/N))
    reward = np.random.randn() + true_values[a]
    N[a] += 1
    Q[a] += (reward - Q[a]) / N[a]

print("UCB:", Q)

# Thompson Sampling
alpha = np.ones(arms)
beta = np.ones(arms)

for t in range(steps):
    samples = [np.random.beta(alpha[i], beta[i]) for i in range(arms)]
    a = np.argmax(samples)
    reward = 1 if random.random() < 0.5 else 0
    alpha[a] += reward
    beta[a] += 1 - reward

print("Thompson Alpha:", alpha)
