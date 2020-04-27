from vector import Vector
from matrix import Matrix

s = 4
v = Vector([2, 1, 0])

m0 = Matrix([[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0]])
m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
m2 = Matrix([[1.0, 0.0], [-1.0, -3.0], [2.0, 1.0]])
m3 = Matrix([[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0]])

print(m0 + m3)
print(m0 - m3)

print(m0 * m1)
print(m0 * s)
print(m2 * v)

print(m0 / s)