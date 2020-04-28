import numpy as np

class MyLinearRegression():
    '''
    Description: My personnal linear regression class to fit like a boss.
    '''

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas


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

    def __gradient(self, x, y, thetas):
        result = [
            self.__forumla(x, y, thetas, 0),
            self.__forumla(x, y, thetas, 1)
        ]
        return np.array(result)

    def __forumla(self, x, y, thetas, j):
        length = x.shape[0]
        value = thetas[j] - self.alpha * (1/length) * np.sum(((x.dot(thetas) - y) * x[:,j:][:,0]))
        return value


    def fit_(self, x, y):
        if self.__isEmpty(x, y) is True:
            return None
        elif self.__dimensionsMatch(x, y) is False:
            return None

        x = self.__addIntercept(self.__parseData(x))
        y = self.__parseData(y)

        t = self.thetas[:]
        
        for i in range(0, self.max_iter):
            oldt = t[:]
            t = self.__gradient(x, y, t)
            if oldt[0] == t[0] and oldt[1] == t[1]:
                setattr(self, 'thetas', t)
                return
        setattr(self, 'thetas', t)
        return


    def predict_(self, x):
        if self.__isEmpty(x) is True:
            return None

        x = self.__addIntercept(self.__parseData(x))

        result = x.dot(self.thetas)
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

