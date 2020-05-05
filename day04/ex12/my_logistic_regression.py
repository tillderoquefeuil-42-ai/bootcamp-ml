import numpy as np

class MyLogisticRegression():

    def __init__(self, thetas, alpha=0.001, n_cycle=1000, penalty='l2', lambda_=0.5):
        self.alpha = alpha
        self.n_cycle = int(n_cycle)
        self.thetas = self.__parseThetas(thetas)
        self.penalty=penalty
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

    def __reshape(self, data):
        if len(data.shape) == 1:
            return data.reshape(data.shape[0], 1)
        return data


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

        x = self.__reshape(x)

        if x.shape[1] != self.thetas.shape[0] - 1:
            return None

        x = self.__addIntercept(x)
        predict = x.dot(self.thetas)

        return 1 / (1 + np.exp(-predict))

    def cost_elem_(self, x, y, eps=1e-15):
        if self.__isEmpty(x, y) is True:
            return None
        elif self.__dimensionsMatch(x, y) is False:
            return None

        y_hat = self.predict_(x)
        m = y.shape[0]
        ones = np.ones((m,1))

        result = (-1/m) * ( y * np.log(y_hat + eps) + (ones - y) * np.log(ones - y_hat + eps) )

        return result

    def cost_(self, x, y):
        result = self.cost_elem_(x, y)
        if result is None:
            return None

        return np.sum(result)

    def fit_(self, x, y, *arg, **kwargs):
        if self.__isEmpty(x, y) is True:
            return None
        x = self.__reshape(x)
        if self.__dimensionsMatch(x, y) is False:
            return None

        alpha = kwargs.get('alpha', self.alpha)
        n_cycle = int(kwargs.get('n_cycle', self.n_cycle))

        x1 = self.__addIntercept(x)
        y = self.__parseData(y)
        l2 = 1 if self.penalty == 'l2' else 0

        m = x1.shape[0]
        for i in range(0, n_cycle):
            oldThetas = self.thetas[:]
            t = self.thetas[:]
            t[0] = 0

            self.thetas = self.thetas - alpha * (1/m) * ( x1.transpose().dot(self.predict_(x) - y) + l2 * self.lambda_ * t)
            if np.array_equal(self.thetas, oldThetas):
                break

        return self.thetas

