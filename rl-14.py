import random

settings = ["Low","Medium","High"]
Q = {s:0 for s in settings}

for _ in range(500):
    action = random.choice(settings)
    reward = random.randint(1,10)  # product quality
    Q[action] += reward

print("Best Machine Setting:", max(Q, key=Q.get))
