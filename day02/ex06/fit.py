import numpy as np


def fit_(x, y, thetas, alpha, n_cycles):

    def __parseThetas(thetas):
        if type(thetas) == list:
            return np.asarray(thetas, dtype=np.float32).reshape(len(thetas), 1)
        elif type(thetas) == np.ndarray and len(thetas) != thetas.shape:
            return thetas.reshape(thetas.shape[0], 1)
        else:
            raise TypeError("thetas has to be -list- or -numpy.ndarray- and contain 2 elements")

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
        return np.concatenate((np.ones(data.shape[0]).reshape(data.shape[0], 1), data), axis=1)

    def __parseData(data):
        return data.reshape(len(data), 1)

    if __isEmpty(x, y) is True:
        return None
    elif __dimensionsMatch(x, y) is False:
        return None

    thetas = __parseThetas(thetas)

    x = __addIntercept(x)
    y = __parseData(y)

    t = thetas[:]
    length = x.shape[0]

    for i in range(0, n_cycles):
        oldt = t[:]

        t = t - (1/length) * alpha * x.transpose().dot((x.dot(t) - y))
        if np.array_equal(t, oldt):
            return t
    return t
