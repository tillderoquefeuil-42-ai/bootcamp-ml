import numpy as np
# import matplotlib.pyplot as plt

class MyLinearRegression():
    '''
    Description: My personnal linear regression class to fit like a boss.
    '''

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = self.__parseThetas(thetas)

    def __parseThetas(self, thetas):
        if type(thetas) == list and len(thetas) > 1:
            return np.array(thetas)
        elif type(thetas) == list and len(thetas) < 2:
            raise ValueError("thetas has contain minimum 2 elements")
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
        return np.insert(data, 0, 1, axis=1)

    def __parseData(self, data):
        if len(data.shape) == 1:
            return data
        if data.shape[0] > data.shape[1]:
            data = data.transpose()
        if len(data.shape) > 1:        
            data = data[0]

        return data

    def __zscore(self, data):
        std = np.std(data, axis=0)
        mean = np.mean(data, axis=0)

        M0 = np.full(data.shape, mean)
        M1 = np.full(data.shape, 1/std)
        return (data - M0) * M1


    def fit_(self, x, y):
        if self.__isEmpty(x, y) is True:
            return None
        elif self.__dimensionsMatch(x, y) is False:
            return None

        x = self.__addIntercept(self.__zscore(x))
        y = self.__parseData(y)

        t = self.thetas[:,0]

        length = x.shape[0]
        for i in range(0, self.max_iter):
            oldt = t[:]

            t = t - (self.alpha/length) * x.transpose().dot((x.dot(t) - y))
            if np.array_equal(t, oldt):
                break

        setattr(self, 'thetas', t.transpose())
        return t

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

        x = self.predict_(x)

        length = x.shape[0]
        value = np.power(x - y, 2)/(2*length)
        return value


    def cost_(self, x, y):
        result = self.cost_elem_(x, y)
        if result is None:
            return None

        return np.sum(result)

    # def plot_best_h(self, x, y):
    #     self.fit_(x, y)

    #     bestFit = self.predict_(x)
    #     plt.plot(x, y, 'co', x, bestFit, 'gx--')
    #     plt.show()

