import math

def cost_(y, y_hat):
    """
    Computes the mean squared error of two non-empty numpy.ndarray, without any for loop.
    The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.ndarray, a vector.
        y_hat: has to be an numpy.ndarray, a vector. 
    Returns:
        The mean squared error of the two vectors as a float. 
        None if y or y_hat are empty numpy.ndarray.
        None if y and y_hat does not share the same dimensions.
    Raises:
        This function should not raise any Exceptions.
    """
    length = len(y)
    if length != len(y_hat):
        return None

    result = 0
    for i in range(0, length):
        value = math.pow((y_hat[i] - y[i]), 2)/(2*length) # remove last '2' to fit the exemple
        result += value

    return result