import numpy as np

def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
        x: has to be an numpy.ndarray, a vector
    Returns:
        The sigmoid value as a numpy.ndarray.
        None if x is an empty numpy.ndarray.
    """

    def __isEmpty(*arg):
        for data in arg:
            if not hasattr(data, 'shape'):
                return True
            elif data.shape[0] == 0:
                return True
        return False

    if __isEmpty(x):
        return None

    return 1 / (1 + np.exp(-x))