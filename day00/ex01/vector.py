

class Vector:
    def __init__(self, arg=None):
        values = None

        if type(arg) is list:
            values = arg
            size = len(values)
        elif type(arg) is tuple:
            values = []
            for x in range(arg[0], arg[1]):
                values.append(float(x))
            size = len(values)
        elif type(arg) is int:
            size = arg
            values = []
            for x in range(0, size):
                values.append(float(x))

        if not values:
            raise ValueError("Must provide minimum one arg")

        self.values = values
        self.size = size

    def __str__(self):
        return "Vector's size: {self.size} - Vector's values : {self.values}".format(self=self)
        
    def __repr__(self):
        return "{{'size':{self.size}, 'values':{self.values}}}".format(self=self)

    def __parseVectors(self, other):
        if not type(other) is Vector:
            raise TypeError("has to be type: -Vector-")
        if self.size != other.size:
            raise ValueError("Can't add different sized vectors")



    def __add__(self, other):
        self.__parseVectors(other)

        values = []
        for x in range(0, self.size):
            values.append(self.values[x] + other.values[x])
        return Vector(values)

    def __radd__(self, other):
        self.__parseVectors(other)

        values = []
        for x in range(0, self.size):
            values.append(other.values[x] + self.values[x])
        return Vector(values)

    def __sub__(self, other):
        self.__parseVectors(other)

        values = []
        for x in range(0, self.size):
            values.append(self.values[x] - other.values[x])
        return Vector(values)

    def __rsub__(self, other):
        self.__parseVectors(other)

        values = []
        for x in range(0, self.size):
            values.append(other.values[x] - self.values[x])
        return Vector(values)

    def __truediv__(self, other):
        values = []

        if isinstance(other, (int, float)):
            for x in range(0, self.size):
                values.append(self.values[x] / other)
            return Vector(values)

        self.__parseVectors(other)

        for x in range(0, self.size):
            values.append(self.values[x] / other.values[x])
        return sum(values)
        # return Vector(sum(values))

    def __rtruediv__(self, other):
        values = []

        if isinstance(other, (int, float)):
            for x in range(0, self.size):
                values.append(other / self.values[x])
            return Vector(values)

        self.__parseVectors(other)

        for x in range(0, self.size):
            values.append(other.values[x] / self.values[x])
        return sum(values)
        # return Vector(sum(values))

    def __mul__(self, other):

        values = []

        if isinstance(other, (int, float)):
            for x in range(0, self.size):
                values.append(self.values[x] * other)
            return Vector(values)

        self.__parseVectors(other)

        for x in range(0, self.size):
            values.append(self.values[x] * other.values[x])
        return sum(values)
        # return Vector(sum(values))

    def __rmul__(self, other):
        values = []

        if isinstance(other, (int, float)):
            for x in range(0, self.size):
                values.append(other * self.values[x])
            return Vector(values)

        self.__parseVectors(other)

        for x in range(0, self.size):
            values.append(other.values[x] * self.values[x])
        return sum(values)
        # return Vector(sum(values))
