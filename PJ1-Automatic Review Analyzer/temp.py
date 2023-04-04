import numpy as np

n = 2
X = np.empty(shape=[0, n])

for i in range(5):
    for j  in range(2):
        X = np.append(X, [[i, j]], axis=0)

print(X)

