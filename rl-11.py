import numpy as np
import random

actions = 9
Q = np.zeros(actions)
N = np.zeros(actions)

def epsilon_greedy(epsilon):
    if random.random() < epsilon:
        return random.randint(0, actions-1)
    return np.argmax(Q)

def softmax(tau=1.0):
    exp_q = np.exp(Q/tau)
    probs = exp_q / np.sum(exp_q)
    return np.random.choice(range(actions), p=probs)

# Simulation
episodes = 500
wins_eps = 0
wins_soft = 0

for _ in range(episodes):
    a = epsilon_greedy(0.1)
    reward = random.choice([0,1])  # 1 = win
    Q[a] += reward
    wins_eps += reward

for _ in range(episodes):
    a = softmax()
    reward = random.choice([0,1])
    wins_soft += reward

print("Epsilon-Greedy Wins:", wins_eps)
print("Softmax Wins:", wins_soft)
