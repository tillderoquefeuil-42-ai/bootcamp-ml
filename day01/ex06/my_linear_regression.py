import numpy as np

class MyLinearRegression():

    def __init__(self, thetas, alpha=0.001, n_cycle=1000):
        self.alpha = alpha
        self.n_cycle = int(n_cycle)
        self.thetas = self.__parseThetas(thetas)

    def __parseThetas(self, thetas):
        if type(thetas) == list:
            return np.asarray(thetas, dtype=np.float32).reshape(2, 1)
        elif type(thetas) == np.ndarray and len(thetas) != thetas.shape:
            return thetas.reshape(thetas.shape[0], 1)
        else:
            raise TypeError("thetas has to be -list- or -numpy.ndarray- and contain 2 elements")

    def __isEmpty(self, *arg):
        for data in arg:
            if not hasattr(data, 'shape'):
                return True
            elif data.shape[0] == 0:
                return True
        return False

    def __dimensionsMatch(self, data0, data1):
        if data0.shape[0] != data1.shape[0]:
            return False
        return True

    def __addIntercept(self, data):
        return np.concatenate((np.ones(data.shape[0]).reshape(data.shape[0], 1), data.reshape((data.shape[0], 1))), axis = 1)

    def __parseData(self, data):
        return data.reshape(len(data), 1)

    def predict_(self, x, *arg, **kwargs):
        thetas = kwargs.get('thetas', self.thetas)

        if self.__isEmpty(x) is True:
            return None

        x = self.__addIntercept(x)

        result = x.dot(thetas)
        return np.array(result)

    def cost_elem_(self, x, y):
        if self.__isEmpty(x, y) is True:
            return None
        elif self.__dimensionsMatch(x, y) is False:
            return None

        x = self.predict_(x)
        y = self.__parseData(y)

        length = y.shape[0]
        value = np.power(x - y, 2) * (1 / (2*length))
        return value

    def cost_(self, x, y):
        result = self.cost_elem_(x, y)
        if result is None:
            return None

        return np.sum(result)

    def gradient(self, x, y):
        y = self.__parseData(y)
        y_hat = self.predict_(x)
        X = self.__addIntercept(x)

        m = len(y)
        Y = y_hat - y

        return (1 / m) * X.transpose().dot(Y)

    def fit_(self, x, y, *arg, **kwargs):
        if self.__isEmpty(x, y) is True:
            return None
        elif self.__dimensionsMatch(x, y) is False:
            return None

        alpha = kwargs.get('alpha', self.alpha)
        n_cycle = int(kwargs.get('n_cycle', self.n_cycle))

        for i in range(n_cycle):
            oldThetas = self.thetas[:]

            grad = self.gradient(x, y)
            self.thetas = self.thetas - alpha * grad
            if np.array_equal(self.thetas, oldThetas):
                break

        return self.thetas
