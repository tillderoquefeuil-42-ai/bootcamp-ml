import numpy as np

def fit_(x, y, theta, alpha, max_iter): 
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem. 
    """
    if x.shape[0] * y.shape[0] * theta.shape[0] == 0:
        return None
    if x.shape[0] != y.shape[0] or theta.shape[0] != 2:
        return None

    x = add_intercept(x.transpose()[0])
    y = y.transpose()[0]

    t = theta[:]
    
    for i in range(0, max_iter):
        oldt = t[:]
        t = gradient(x, y, t, alpha)
        if oldt[0] == t[0] and oldt[1] == t[1]:
            return t
    return t


def gradient(x, y, theta, alpha):
    if x.shape[0] * y.shape[0] * theta.shape[0] == 0:
        return None
    if x.shape[0] != y.shape[0] or theta.shape[0] != 2:
        return None

    result = [
        forumla(x, y, theta, alpha, 0),
        forumla(x, y, theta, alpha, 1)
    ]
    return np.array(result)

def forumla(x, y, theta, alpha, j):
    length = x.shape[0]
    value = theta[j] - alpha * (1/length) * sum(((x.dot(theta) - y) * x[:,j:][:,0]))
    return value

def add_intercept(x):
    if x.shape[0] == 0:
        return None

    if 1 == len(x.shape):
        x = x.reshape(x.shape[0], 1)

    return np.insert(x, 0, 1, axis=1)