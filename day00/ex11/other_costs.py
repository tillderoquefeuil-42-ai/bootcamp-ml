import math

def mse_(y, y_hat):
    """
    Description:
    Calculate the MSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.ndarray, a vector of dimension m * 1. 
        y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
    Returns:
        mse: has to be a float.
        None if there is a matching dimension problem.
    """
    length = len(y)
    if length != len(y_hat):
        return None

    result = 0
    for i in range(0, length):
        value = math.pow((y_hat[i] - y[i]), 2)/length
        result += value

    return result

def rmse_(y, y_hat):
    """
    Description:
        Calculate the RMSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        y_hat: has to be a numpy.ndarray, a vector of dimension m * 1. 
    Returns:
        rmse: has to be a float.
        None if there is a matching dimension problem.
    """
    length = len(y)
    if length != len(y_hat):
        return None
    
    return math.sqrt(mse_(y, y_hat))

def mae_(y, y_hat):
    """
    Description:
        Calculate the MAE between the predicted output and the real output.
    Args:
        y: has to be a numpy.ndarray, a vector of dimension m * 1. 
        y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
    Returns:
        mae: has to be a float.
        None if there is a matching dimension problem.
    """
    length = len(y)
    if length != len(y_hat):
        return None

    result = 0
    for i in range(0, length):
        value = abs(y_hat[i] - y[i])/length
        result += value

    return result

def r2score_(y, y_hat):
    """
    Description:
    C   alculate the R2score between the predicted output and the output.
    Args:
        y: has to be a numpy.ndarray, a vector of dimension m * 1. 
        y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
    Returns:
        r2score: has to be a float.
        None if there is a matching dimension problem.
    """
    length = len(y)
    if length != len(y_hat):
        return None

    result = [0, 0]
    y_av = sum(y)/len(y)
    for i in range(0, length):
        result[0] += math.pow((y_hat[i] - y[i]), 2)
        result[1] += math.pow((y_hat[i] - y_av), 2)
    
    return (1 - result[0]/result[1])