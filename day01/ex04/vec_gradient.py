import numpy as np

def gradient(x, y, theta):
    """
    Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop.
    The three arrays must have compatible dimensions. 
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1. 
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a 2 * 1 vector. 
    Returns:
        The gradient as a numpy.ndarray, a vector of dimension 2 * 1. 
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    """
    if x.shape[0] * y.shape[0] * theta.shape[0] == 0:
        return None
    if x.shape[0] != y.shape[0] or theta.shape[0] != 2:
        return None

    x = add_intercept(x)

    result = [
        forumla(x, y, theta, 0),
        forumla(x, y, theta, 1)
    ]
    return result

def forumla(x, y, theta, j):
    length = x.shape[0]
    return theta[j] - (1/length) * sum(((x.dot(theta) - y) * x[:,j:][:,0]))

def add_intercept(x):
    if x.shape[0] == 0:
        return None

    if 1 == len(x.shape):
        x = x.reshape(x.shape[0], 1)

    return np.insert(x, 0, 1, axis=1)