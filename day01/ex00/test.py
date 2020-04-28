import numpy as np
from cost import cost_, cost_elem_
from prediction import predict_

x1 = np.array([[0.], [1.], [2.], [3.], [4.]]) 
theta1 = np.array([[2.], [4.]])

y_hat1 = predict_(x1, theta1)
print(y_hat1)

y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

# Example 1:
print(cost_elem_(y1, y_hat1))
# Output:
# array([[0.], [0.1], [0.4], [0.9], [1.6]]) 

# Example 2:
print(cost_(y1, y_hat1))
# Output:
# 3.0

x2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]]) 
theta2 = np.array([[0.05], [1.], [1.], [1.]])
y_hat2 = predict_(x2, theta2)
y2 = np.array([[19.], [42.], [67.], [93.]])

# Example 3:
print(cost_elem_(y2, y_hat2))
# Output:
# array([[1.3203125], [0.7503125], [0.0153125], [2.1528125]])

# Example 4:
print(cost_(y2, y_hat2))
# Output:
# 4.238750000000004

x3 = np.array([0, 15, -9, 7, 12, 3, -21])
theta3 = np.array([[0.], [1.]])
y_hat3 = predict_(x3, theta3)
y3 = np.array([2, 14, -13, 5, 12, 4, -19])

# Example 5:
print(cost_(y3, y_hat3))
# Output:
# 4.285714285714286
# Example 6:
print(cost_(y3, y3))
# Output:
# 0.0