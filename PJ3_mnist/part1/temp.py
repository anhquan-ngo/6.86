import numpy as np
filter = np.array([[1,0.5],[0.5,1]])
I_1 = np.array([[1,2],[2,1]])
I_2 = np.array([[2,1],[1,1]])
I_3 = np.array([[2,1],[1,1]])
I_4 = np.array([[1,1],[1,1]])
list = [I_1, I_2, I_3, I_4]
for l in list:
    print(np.inner(l,filter))