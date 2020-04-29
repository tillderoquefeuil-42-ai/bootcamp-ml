import numpy as np

def gradient(x, y, theta):

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

    if __isEmpty(x, y) is True:
        return None
    elif __dimensionsMatch(x, y) is False:
        return None

    x = __addIntercept(x)

    result = (1/x.shape[0]) * x.transpose().dot((x.dot(theta) - y))
    return np.array(result)


    

