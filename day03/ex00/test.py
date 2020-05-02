import numpy as np
from mylinearregression import MyLinearRegression as MyLR

X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
Y = np.array([[23.], [48.], [218.]])

alpha = 0.0001
n_cycle = 160000
mylr = MyLR([[1.], [1.], [1.], [1.], [1]], alpha=alpha, n_cycle=n_cycle)
# mylr = MyLR([[0.], [0.], [0.], [0.], [0]])

# Example 0:
print(mylr.predict_(X))
# Output:
# array([[8.], [48.], [323.]])

# Example 1:
print(mylr.cost_elem_(X,Y))
# Output:
# array([[37.5], [0.], [1837.5]])

# Example 2:
print(mylr.cost_(X,Y))
# Output: 
# 1875.0

# Example 3:
mylr.fit_(X, Y)
print(mylr.thetas)
# Output:
# array([[18.023..], [3.323..], [-0.711..], [1.605..], [-0.1113..]])

# Example 4:
print(mylr.predict_(X))
# Output:
# array([[23.499..], [47.385..], [218.079...]])

# Example 5:
print(mylr.cost_elem_(X,Y))
# Output:
# array([[0.041..], [0.062..], [0.001..]])

# Example 6:
print(mylr.cost_(X,Y))
# Output: 
# 0.1056..