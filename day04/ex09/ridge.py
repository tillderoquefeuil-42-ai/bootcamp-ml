import numpy as np

class MyRidge():
    """
    Description:
    My personnal ridge regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, n_cycle=1000, lambda_=0.5):
        self.alpha = alpha
        self.n_cycle = int(n_cycle)
        self.thetas = self.__parseThetas(thetas)
        self.lambda_ = lambda_

    def __parseThetas(self, thetas):
        if type(thetas) == list:
            return np.asarray(thetas, dtype=np.float32).reshape(len(thetas), 1)
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
        return np.concatenate((np.ones(data.shape[0]).reshape(data.shape[0], 1), data), axis=1)

    def __parseData(self, data):
        return data.reshape(len(data), 1)

    def predict_(self, x):
        if self.__isEmpty(x) is True:
            return None

        x = self.__addIntercept(x)
        result = x.dot(self.thetas)

        return result

    def cost_elem_(self, x, y):
        if self.__isEmpty(x, y) is True:
            return None
        elif self.__dimensionsMatch(x, y) is False:
            return None

        m = y.shape[0]
        t = self.thetas[:]
        t[0] = 0

        y_hat = self.predict_(x)

        cost = y_hat - y
        result = 1/(2*m) * ( cost.dot(cost) + self.lambda_ * t.dot(t) )
        return result

    def cost_(self, x, y):
        result = self.cost_elem_(x, y)
        if result is None:
            return None

        return np.sum(result)

    def fit_(self, x, y, *arg, **kwargs):
        if self.__isEmpty(x, y) is True:
            return None
        elif self.__dimensionsMatch(x, y) is False:
            return None

        alpha = kwargs.get('alpha', self.alpha)
        n_cycle = int(kwargs.get('n_cycle', self.n_cycle))

        x = self.__addIntercept(x)
        y = self.__parseData(y)

        m = x.shape[0]

        for i in range(0, n_cycle):
            oldThetas = self.thetas[:]

            t = self.thetas[:]
            t[0] = 0

            cost = x.dot(self.thetas) - y
            self.thetas = self.thetas - alpha * (1/m) * (x.T.dot(cost) + self.lambda_ * t)
            if np.array_equal(self.thetas, oldThetas):
                break

        return self.thetas

    def mse_(self, x, y):
        if self.__isEmpty(x, y) is True:
            return None
        elif self.__dimensionsMatch(x, y) is False:
            return None

        x = self.predict_(x)

        length = x.shape[0]
        value = np.power(x - y, 2)/(length)
        return np.sum(value)
