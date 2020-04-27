from vector import Vector

print("TEST VECTOR INIT\n")

try:
    v0 = Vector()
    print(v0)
except ValueError as error:
    print("ERROR: " + str(error))

try:
    v1 = Vector(2)
    print(v1)
except ValueError as error:
    print("ERROR: " + str(error))

try:
    v2 = Vector([1.0, 2.0, 3.0, 4.0])
    print(v2)
except ValueError as error:
    print("ERROR: " + str(error))

try:
    v3 = Vector((10, 15))
    print(v3)
except ValueError as error:
    print("ERROR: " + str(error))


print("\n\nTEST VECTOR OPERATIONS\n")

v4 = Vector((5, 10))
print(v4)

try:
    print(v1 + 4)
except (ValueError, TypeError) as error:
    print("ERROR: " + str(error))

try:
    print(v1 - v2)
except (ValueError, TypeError) as error:
    print("ERROR: " + str(error))

print(v3 + v4)
print(v3 - v4)
print(v3 * v4)
print(v3 * 5)
print(v3 / v4)
print(v3 / 5)