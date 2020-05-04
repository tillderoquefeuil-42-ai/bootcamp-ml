
def __isEmpty(*arg):
    for data in arg:
        if not hasattr(data, 'shape'):
            return True
        elif data.shape[0] == 0:
            return True
    return False

def iterative_l2(theta):
    """
    Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of dimension n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    """
    if __isEmpty(theta):
        return None

    result = 0
    for t in theta[1:]:
        result += t*t

    return result

def l2(theta):
    """
    Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of dimension n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    """
    if __isEmpty(theta):
        return None

    theta[0] = 0
    return theta.dot(theta)
