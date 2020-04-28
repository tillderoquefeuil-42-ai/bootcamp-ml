import numpy as np

def minmax(x):
    """
    Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
    Args:
        x: has to be an numpy.ndarray, a vector. 
    Returns:
        x' as a numpy.ndarray.
        None if x is a non-empty numpy.ndarray.
    """
    if not hasattr(x, 'shape'):
        return None
    elif x.shape[0] == 0:
        return None

    minimum = min(x)
    maximum = max(x)

    M0 = np.full(x.shape, minimum)
    M1 = np.full(x.shape, 1/(maximum-minimum))

    return (x - M0) * M1

