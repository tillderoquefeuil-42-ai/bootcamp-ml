import numpy as np
from log_pred import logistic_predict_

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

def reshape(data):
    if len(data.shape) == 1:
        return data.reshape(data.shape[0], 1)
    return data



def log_gradient(x, y, theta):
    """
    Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop. 
    The three arrays must have compatible dimensions.
    Args:
        x: has to be an numpy.ndarray, a matrix of dimension m * n.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector (n +1) * 1.
    Returns:
        The gradient as a numpy.ndarray, a vector of dimensions n * 1, containing the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    """

    if __isEmpty(x, y) is True:
        return None
    x = reshape(x)
    if __dimensionsMatch(x, y) is False:
        return None

    y_hat = logistic_predict_(x, theta)

    x = __addIntercept(x)
    y = __parseData(y)

    m = x.shape[0]

    return (1/m) * x.transpose().dot(y_hat - y)
