import numpy as np

def accuracy_score_(y, y_hat):
    """
    Compute the accuracy score. 
    Args:
        y:a numpy.ndarray for the correct labels 
        y_hat:a numpy.ndarray for the predicted labels
    Returns:
        The accuracy score as a float.
        None on any error.
    """    
    return np.sum(y == y_hat)/y.shape[0]


def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
        The precision score as a float.
        None on any error.
    """
    classLabel = np.full(y.shape, pos_label)

    a = classLabel == y_hat
    b = y == y_hat
    return np.sum(a * b)/np.sum(a)

def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
        The recall score as a float.
        None on any error.
    """
    classLabel = np.full(y.shape, pos_label)

    a = classLabel == y_hat
    b = y == y_hat
    return np.sum(a * b)/np.sum(classLabel == y)


def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.    
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
        The f1 score as a float.
        None on any error.
    """
    classLabel = np.full(y.shape, pos_label)

    a = classLabel == y_hat
    b = y == y_hat

    p = np.sum(a * b)/np.sum(a)
    r = np.sum(a * b)/np.sum(classLabel == y)
    return 2 * (p * r)/(p + r)

