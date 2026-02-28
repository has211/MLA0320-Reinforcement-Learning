import numpy as np

gamma = 0.9
grid = 4
V = np.zeros((grid,grid))

for _ in range(100):
    for i in range(grid):
        for j in range(grid):
            V[i,j] = 1 + gamma * max(
                V[i-1,j] if i>0 else 0,
                V[i+1,j] if i<grid-1 else 0,
                V[i,j-1] if j>0 else 0,
                V[i,j+1] if j<grid-1 else 0
            )

print("Optimal Value Function:\n", V)
