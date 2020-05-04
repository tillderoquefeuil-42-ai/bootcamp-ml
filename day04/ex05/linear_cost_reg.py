import numpy as np

def reg_cost_(y, y_hat, theta, lambda_):
    """
    Computes the regularized cost of a linear regression model 
    from two non-empty numpy.ndarray, without any for loop. 
    The two arrays must have the same dimensions. 
    Args:
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        y_hat: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be a numpy.ndarray, a vector of dimension n * 1.
        lambda_: has to be a float.
    Returns:
        The regularized cost as a float.
        None if y, y_hat, or theta are empty numpy.ndarray. 
        None if y and y_hat do not share the same dimensions.
    """

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

    if __isEmpty(y_hat, y):
        return None
    elif __dimensionsMatch(y_hat, y) is False:
        return None

    m = y.shape[0]
    t = theta[:]
    t[0] = 0

    cost = y_hat - y
    result = 1/(2*m) * ( cost.dot(cost) + lambda_ * t.dot(t) )
    return np.sum(result)
