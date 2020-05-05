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


def reg_logistic_grad(y, x, theta, lambda_):
    """
    Computes the regularized logistic gradient of three non-empty numpy.ndarray, with two for-loop. 
    The three arrays must have compatible dimensions. 
    Args:
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n. 
        theta: has to be a numpy.ndarray, a vector of dimension n * 1. 
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of dimension n * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles dimensions.
    """
    if __isEmpty(x, y):
        return None
    elif __dimensionsMatch(x, y) is False:
        return None

    x = __addIntercept(x)
    m = x.shape[0]

    t = theta[:]
    t[0] = 0

    predict = 1 / (1 + np.exp(-x.dot(theta)))
    cost = predict - y

    result = []
    for j in range(0, len(theta)):
        value = 0
        if j == 0:
            value = (1/m) * cost
        else:
            value = (1/m) * ( np.sum(x.T[j].dot(cost)) + (lambda_ * t[j][0]) )
        result.append(np.sum(value))

    result = np.array([result]).T
    return result



def vec_reg_logistic_grad(y, x, theta, lambda_):
    """
    Computes the regularized logistic gradient of three non-empty numpy.ndarray, without any for-loop.
    The three arrays must have compatible dimensions. 
    Args:
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n. 
        theta: has to be a numpy.ndarray, a vector of dimension n * 1. 
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of dimension n * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles dimensions.
    """
    if __isEmpty(x, y):
        return None
    elif __dimensionsMatch(x, y) is False:
        return None

    X = __addIntercept(x)
    m = x.shape[0]

    t = theta[:]
    t[0] = 0

    predict = 1 / (1 + np.exp(-X.dot(theta)))
    cost = predict - y

    return (1/m) * (X.T.dot(cost) + lambda_ * t)

