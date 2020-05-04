import numpy as np

def add_polynomial_features(x, power):
    """
    Add polynomial features to matrix x by raising its columns 
    to every power in the range of 1 up to the power given in argument.
    Args:
        x: has to be an numpy.ndarray, a matrix of dimension m * n.
        power: has to be an int, the power up to which the columns of matrix x are going to be raised.
    Returns:
        The matrix of polynomial features as a numpy.ndarray, of dimension m * (np), 
        containg the polynomial feature values for all training examples. 
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
    
    data = x[:]
    for i in range (2, power+1):
        data = np.concatenate((data, np.power(x, i)), axis=1)
    return data