import numpy as np
from vec_cost import cost_

X = np.array([0, 15, -9, 7, 12, 3, -21]) 
Y = np.array([2, 14, -13, 5, 12, 4, -19])

# Example 1:
print(cost_(X, Y))
# Output: 
# 4.285714285714286

# Example 2:
print(cost_(X, X))
# Output: 
# 0.0