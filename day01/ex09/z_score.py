import numpy as np

def zscore(x):
    """
    Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization. 
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

    std = np.std(x)
    mean = np.mean(x)

    M0 = np.full(x.shape, mean)
    M1 = np.full(x.shape, 1/std)

    return (x - M0) * M1

