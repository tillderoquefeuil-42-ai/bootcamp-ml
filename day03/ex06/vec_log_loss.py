import numpy as np

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

def vec_log_loss_(y, y_hat, eps=1e-15):
    if __isEmpty(y, y_hat) is True:
        return None
    elif __dimensionsMatch(y, y_hat) is False:
        return None

    m = y.shape[0]

    ones = np.ones((m,1))
    result = (-1/m) * ( y * np.log(y_hat) + (ones - y) * np.log(ones - y_hat) )
    # result = (-1/m) * ( y * np.log(y_hat + eps) + (ones - y) * np.log(ones - y_hat + eps) )

    return np.sum(result)
