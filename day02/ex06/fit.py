import numpy as np


def fit_(x, y, theta, alpha, n_cycles):

    def __isEmpty(*arg):
        for data in arg:
            if not hasattr(data, 'shape'):
                return True
            elif data.shape[0] == 0:
                return True
        return False

    def __dimensionsMatch(data0, data1):
        if data0.shape[0] != data1.shape[0]:
            return False
        return True

    def __addIntercept(data):
        return np.insert(data, 0, 1, axis=1)

    def __parseTheta(thetas):
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

    def __parseData(data):
        if len(data.shape) == 1:
            return data
        if data.shape[0] > data.shape[1]:
            data = data.transpose()
        if len(data.shape) > 1:        
            data = data[0]
        return data        

    if __isEmpty(x, y) is True:
        return None
    elif __dimensionsMatch(x, y) is False:
        return None

    theta = __parseTheta(theta)

    x = __addIntercept(x)
    y = __parseData(y)

    t = theta[:]
    length = x.shape[0]
    for i in range(0, n_cycles):
        oldt = t[:]

        t = t - (1/length) * alpha * x.transpose().dot((x.dot(t) - y))
        if np.array_equal(t, oldt):
            return t
    return t
