import numpy as np

def cost_(y, y_hat):

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


    if __isEmpty(y, y_hat) is True:
        return None
    elif __dimensionsMatch(y, y_hat) is False:
        return None

    length = y.shape[0]
    result = np.power(y_hat - y, 2)/length

    return np.sum(result)