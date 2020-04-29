import numpy as np
import matplotlib.pyplot as plt

class MyLinearRegression():
    '''
    Description: My personnal linear regression class to fit like a boss.
    '''

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = self.__parseThetas(thetas)

    def __parseThetas(self, thetas):
        if type(thetas) == list and len(thetas) == 2:
            return np.array(thetas)
        elif type(thetas) == list and len(thetas) != 2:
            raise ValueError("thetas has contain 2 elements")
        elif type(thetas) == np.ndarray and len(thetas.shape) == 2:
            return thetas[:,0]
        elif type(thetas) == np.ndarray and len(thetas.shape) == 1:
            return thetas
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
        data = data.reshape(data.shape[0], 1)
        return np.insert(data, 0, 1, axis=1)

    def __parseData(self, data):
        if len(data.shape) == 1:
            return data
        if data.shape[0] > data.shape[1]:
            data = data.transpose()
        if len(data.shape) > 1:        
            data = data[0]

        return data

    def __gradient(self, x, y, thetas, alpha):
        result = [
            self.__forumla(x, y, thetas, 0, alpha),
            self.__forumla(x, y, thetas, 1, alpha)
        ]
        return np.array(result)

    def __forumla(self, x, y, thetas, j, alpha):
        length = x.shape[0]
        value = thetas[j] - alpha * (1/length) * np.sum(((x.dot(thetas) - y) * x[:,j:][:,0]))
        return value


    def fit_(self, x, y, *arg, **kwargs):
        if self.__isEmpty(x, y) is True:
            return None
        elif self.__dimensionsMatch(x, y) is False:
            return None

        thetas = kwargs.get('thetas', self.thetas)
        alpha = kwargs.get('alpha', self.alpha)


        x = self.__addIntercept(self.__parseData(x))
        y = self.__parseData(y)

        t_values = []
        t = thetas[:]
        for i in range(0, self.max_iter):
            t_values.append(t)
            oldt = t[:]
            t = self.__gradient(x, y, t, alpha=alpha)
            if oldt[0] == t[0] and oldt[1] == t[1]:
                setattr(self, 'thetas', t)
                return t_values
        setattr(self, 'thetas', t)
        return t_values


    def predict_(self, x, *arg, **kwargs):
        thetas = kwargs.get('thetas', self.thetas)

        if self.__isEmpty(x) is True:
            return None

        x = self.__addIntercept(self.__parseData(x))

        result = x.dot(thetas)
        return np.array(result)


    def cost_elem_(self, x, y):
        if self.__isEmpty(x, y) is True:
            return None
        elif self.__dimensionsMatch(x, y) is False:
            return None

        x = self.__parseData(x)
        y = self.__parseData(y)

        length = x.shape[0]
        value = np.power(x - y, 2)/length
        return value


    def cost_(self, x, y):
        result = self.cost_elem_(x, y)
        if result is None:
            return None

        return np.sum(result)

    def plot_best_h(self, x, y):
        self.fit_(x, y)

        bestFit = self.predict_(x)
        plt.plot(x, y, 'co', x, bestFit, 'gx--')
        plt.show()

