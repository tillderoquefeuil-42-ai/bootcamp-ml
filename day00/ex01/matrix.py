from vector import Vector

class Matrix:
    def __init__(self, arg=None):
        data = None

        if type(arg) is list and all(isinstance(x, list) for x in arg):
            data = arg
            shape = self.__parseMatrix(data)
        elif type(arg) is tuple:
            shape = arg
            data = []
            for x in range(0, shape[0]):
                column = []
                for y in range(0, shape[1]):
                    column.append(float(x))
                data.append(column)

        if type(arg) is list and not all(isinstance(x, list) for x in arg):
            raise TypeError("Must contain -list- only")
        elif arg is None:
            raise ValueError("Must provide minimum one arg")
        elif not data:
            raise TypeError("First argument has to be type -list- or -tuple-")

        self.data = data
        self.shape = shape

    def __str__(self):
        return "Matrix's dimensions are {self.shape} - Matrix's data are {self.data}".format(self=self)

    def __repr__(self):
        return "{{'shape':{self.shape}, 'data':{self.data}}}".format(self=self)


    def __parseMatrix(self, data):
        rows = len(data)

        columns = None
        for x in data:
            if not all(isinstance(y, (int, float)) for y in x):
                raise TypeError("Must contain -int- or -float- only")
            if columns is None:
                columns = len(x)
            elif columns != len(x):
                raise ValueError("List of different size")

        return (columns, rows)

    def __checkMatrixType(self, other):
        if not type(other) is Matrix:
            raise TypeError("has to be type: -Matrix-")

    def __compareMatricesShapes(self, other, *arg, **kwargs):
        strict = kwargs.get('strict', True)
        reverse = kwargs.get('reverse', False)

        if strict is True and self.shape != other.shape:
            raise ValueError("Can't operate with different shaped Matrices")
        elif strict is False:
            if reverse is False and self.shape[0] != other.shape[1]:
                raise ValueError("Can't operate with different shaped Matrices")
            if reverse is True and self.shape[1] != other.shape[0]:
                raise ValueError("Can't operate with different shaped Matrices")

    def __compareMatrixVectorShapes(self, vector):
        if self.shape[1] != vector.size:
            raise ValueError("Can't operate with different shaped Matrix and Vector")


    def __numeric(self, a, b, operator):
        if operator == '+':
            return a + b
        elif operator == '-':
            return a - b
        elif operator == '*':
            return a * b
        elif operator == '/':
            return a / b
        return None

    def __getEmptyMatrix(self, matrix0, matrix1):
        data = []
        for x in range(0, matrix0.shape[1]):
            column = []
            for y in range(0, matrix1.shape[0]):
                column.append(0)
            data.append(column)
        return data

    def __matrixAndScalar(self, scalar, operator):
        data = []
        for x in range(0, self.shape[1]):
            column = []
            for y in range(0, self.shape[0]):
                value = self.__numeric(self.data[x][y], scalar, operator)
                column.append(value)
            data.append(column)
        return Matrix(data)

    def __matrixAndVector(self, vector, operator):
        self.__compareMatrixVectorShapes(vector)

        data = []
        for y in range(0, self.shape[0]):
            value = 0
            for x in range(0, self.shape[1]):
                value += self.__numeric(self.data[x][y], vector.values[x], operator)
            data.append(value)

        return Vector(data)

    def __matrixAndMatrix(self, matrix, operator, *arg, **kwargs):
        reverse = kwargs.get('reverse', False)

        self.__compareMatricesShapes(matrix, strict=False, reverse=reverse)

        matrix0 = self if reverse is False else matrix
        matrix1 = self if reverse is True else matrix

        data = self.__getEmptyMatrix(matrix0, matrix1)

        for i in range(0, matrix0.shape[1]):
            for j in range(0, matrix1.shape[0]):
                for k in range(0, matrix1.shape[1]):
                    value = self.__numeric(matrix0.data[i][k], matrix1.data[k][j], operator)
                    data[i][j] += value

        return Matrix(data)


    def __add__(self, other):
        self.__checkMatrixType(other)
        self.__compareMatricesShapes(other, strict=True)

        data = []
        for x in range(0, self.shape[1]):
            column = []
            for y in range(0, self.shape[0]):
                column.append(self.data[x][y] + other.data[x][y])
            data.append(column)

        return Matrix(data)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        self.__checkMatrixType(other)
        self.__compareMatricesShapes(other, strict=True)

        data = []
        for x in range(0, self.shape[1]):
            column = []
            for y in range(0, self.shape[0]):
                column.append(self.data[x][y] - other.data[x][y])
            data.append(column)

        return Matrix(data)

    def __rsub__(self, other):
        self.__checkMatrixType(other)
        self.__compareMatricesShapes(other, strict=True)

        data = []
        for x in range(0, self.shape[1]):
            column = []
            for y in range(0, self.shape[0]):
                column.append(other.data[x][y] - self.data[x][y])
            data.append(column)

        return Matrix(data)


    def __truediv__(self, other):
        # SCALAR
        if isinstance(other, (int, float)):
            return self.__matrixAndScalar(other, '/')
        else:
            raise TypeError("Type has to be -int-, -float-, -Vector-")

    def __rtruediv__(self, other):
        # SCALAR
        if isinstance(other, (int, float)):
            return self.__matrixAndScalar(other, '/')
        else:
            raise TypeError("Type has to be -int-, -float-, -Vector-")

    def __mul__(self, other):
        # SCALAR
        if isinstance(other, (int, float)):
            return self.__matrixAndScalar(other, '*')
        # VECTOR
        elif isinstance(other, (Vector)):
            return self.__matrixAndVector(other, '*')
        # MATRIX
        elif isinstance(other, (Matrix)):
            return self.__matrixAndMatrix(other, '*')
        else:
            raise TypeError("Type has to be -int-, -float-, -Vector- or -Matrix-")

    def __rmul__(self, other):
        # SCALAR
        if isinstance(other, (int, float)):
            return self.__matrixAndScalar(other, '*')
        # VECTOR
        elif isinstance(other, (Vector)):
            return self.__matrixAndVector(other, '*')
        # MATRIX
        elif isinstance(other, (Matrix)):
            return self.__matrixAndMatrix(other, '*', reverse=True)
        else:
            raise TypeError("Type has to be -int-, -float-, -Vector- or -Matrix-")

