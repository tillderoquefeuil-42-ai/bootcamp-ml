import numpy as np

def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray. 
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1. 
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exception.
    """
    if x.shape[0] == 0 or theta.shape[0] == 0:
        return None

    result = []
    for i in x:
        result.append(theta[0] + theta[1] * i)

    return np.array(result)