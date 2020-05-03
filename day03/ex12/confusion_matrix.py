import numpy as np

def prac(y_true, y_hat, l1, l2):
    cl1 = np.full(y.shape, l1)
    cl2 = np.full(y.shape, l2)

    a = cl1 == y_true
    b = cl2 == y_hat
    return np.sum(a * b)

def confusion_matrix_(y_true, y_hat, labels=None):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        labels: optional, a list of labels to index the matrix. This may be used to reorder or select a subset of labels. (default=None)
    Returns:
        The confusion matrix as a numpy ndarray.
        None on any error.
    """

    if labels is None:
        labels = np.unique(np.concatenate((y_true,y_hat))).tolist()

    result = []
    for l1 in labels:
        row = []
        for l2 in labels:
            row.append(prac(y_true, y_hat, l1, l2))
        result.append(row)

    return np.array(result)