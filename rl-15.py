import random

returns = []

for episode in range(1000):
    churn = random.choice([0,1])  # 1=customer churned
    reward = 1 if churn==0 else -1
    returns.append(reward)

value = sum(returns)/len(returns)
print("Estimated Policy Value:", value)
