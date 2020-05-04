import numpy as np
from my_logistic_regression import MyLogisticRegression as MyLR

X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
Y = np.array([[1], [0], [1]])

# mylr = MyLR([2, 0.5, 7.1, -4.3, 2.09])
mylr = MyLR([0., 0., 0., 0., 0.])

# Example 0:
print(mylr.predict_(X))
# Output:
# array([[0.99930437], [1. ],[1. ]])

# Example 1:
print(mylr.cost_(X,Y))
# Output: 11.513157421577004

# Example 2:
mylr.fit_(X, Y, alpha=1e-4, n_cycle=50000)
print(mylr.thetas)
# exit()
# Output:
# array([[ 1.04565272], [ 0.62555148], [ 0.38387466], [ 0.15622435], [-0.45990099]])

# Example 3:
print(mylr.predict_(X))
# Output:
# array([[0.72865802], [0.40550072], [0.45241588]])

# Example 4:
print(mylr.cost_(X,Y))
# Output:
# 0.5432466580663214