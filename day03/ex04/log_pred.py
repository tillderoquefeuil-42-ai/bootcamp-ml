import numpy as np

def __isEmpty(*arg):
    for data in arg:
        if not hasattr(data, 'shape'):
            return True
        elif data.shape[0] == 0:
            return True
    return False

def reshape(data):
    if len(data.shape) == 1:
        return data.reshape(data.shape[0], 1)
    return data

def addIntercept(data):
    return np.concatenate((np.ones(data.shape[0]).reshape(data.shape[0], 1), data), axis=1)

def sigmoid_(x):
    if __isEmpty(x):
        return None

    return 1 / (1 + np.exp(-x))

def logistic_predict_(x, theta):
    """
    Computes the vector of prediction y_hat from two non-empty numpy.ndarray. 
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1. 
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    """
    if __isEmpty(x, theta):
        return None

    x = reshape(x)

    if x.shape[1] != theta.shape[0] - 1:
        return None

    x = addIntercept(x)

    predic = x.dot(theta)
    return sigmoid_(predic)

    