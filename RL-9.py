import random

returns = []

for _ in range(1000):
    reward = random.randint(1,5)
    returns.append(reward)

print("Estimated Value:", sum(returns)/len(returns))
