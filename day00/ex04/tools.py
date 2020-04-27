import numpy as np

def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.ndarray x. 
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
    Returns:
        X as a numpy.ndarray, a vector of dimension m * 2. None if x is not a numpy.ndarray.
        None if x is a empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if x.shape[0] == 0:
        return None

    if 1 == len(x.shape):
        x = x.reshape(x.shape[0], 1)

    return np.insert(x, 0, 1, axis=1)
