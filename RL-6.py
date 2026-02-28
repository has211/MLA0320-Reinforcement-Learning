import numpy as np
import random

true_ctr = [0.05, 0.1, 0.15]
estimate = np.zeros(3)
count = np.zeros(3)

for _ in range(1000):
    ad = np.argmax(estimate)
    click = 1 if random.random() < true_ctr[ad] else 0
    count[ad] += 1
    estimate[ad] += (click - estimate[ad]) / count[ad]

print("Best Ad:", np.argmax(estimate))
