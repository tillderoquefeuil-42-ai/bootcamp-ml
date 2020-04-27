import math
import numpy as np


def reshape(data):
    if 1 == len(data.shape):
        data = data.reshape(data.shape[0], 1)
    return data


def cost_elem_(y, y_hat): 
    """
    Description:
        Calculates all the elements (1/2*M)*(y_pred - y)^2 of the cost function.
    Args:
        y: has to be an numpy.ndarray, a vector. 
        y_hat: has to be an numpy.ndarray, a vector.
    Returns:
        J_elem: numpy.ndarray, a vector of dimension (number of the training examples,1).
        None if there is a dimension matching problem between X, Y or theta.
    Raises:
        This function should not raise any Exception.
    """
    length = len(y)
    if length != len(y_hat):
        return None

    y = reshape(y)
    y_hat = reshape(y_hat)

    result = []
    for i in range(0, length):
        value = math.pow((y_hat[i][0] - y[i][0]), 2)/(2*length)
        result.append([value])

    return result

def cost_(y, y_hat): 
    """
    Description:
        Calculates the value of cost function.
    Args:
        y: has to be an numpy.ndarray, a vector. 
        y_hat: has to be an numpy.ndarray, a vector.
    Returns:
        J_value : has to be a float.
        None if there is a dimension matching problem between X, Y or theta. 
    Raises:
        This function should not raise any Exception.
    """
    length = len(y)
    if length != len(y_hat):
        return None

    y = reshape(y)
    y_hat = reshape(y_hat)

    result = 0
    for i in range(0, length):
        value = math.pow((y_hat[i][0] - y[i][0]), 2)/(2*length)
        result += value

    return result
