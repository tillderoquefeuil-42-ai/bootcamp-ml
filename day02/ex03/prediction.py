import numpy as np

def predict_(x, theta):

    def __isEmpty(*arg):
        for data in arg:
            if not hasattr(data, 'shape'):
                return True
            elif data.shape[0] == 0:
                return True
        return False

    def __addIntercept(data):
        return np.insert(data, 0, 1, axis=1)

    if __isEmpty(x) is True:
        return None

    x = __addIntercept(x)
    result = x.dot(theta)

    return np.array(result)

