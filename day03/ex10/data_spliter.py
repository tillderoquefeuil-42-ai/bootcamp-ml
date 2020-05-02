import numpy as np

def data_spliter(x, y, proportion):
    """
    Shuffles and splits the dataset (given by x and y) 
    into a training and a test set,
    while respecting the given proportion of examples to be kept in the traning set. 
    Args:
        x: has to be an numpy.ndarray, a matrix of dimension m * n.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        proportion: has to be a float, the proportion of the dataset 
            that will be assigned to the training set.
    Returns:
        (x_train, x_test, y_train, y_test) as a tuple of numpy.ndarray 
        None if x or y is an empty numpy.ndarray.
        None if x and y do not share compatible dimensions.
    """

    def __isEmpty(*arg):
        for data in arg:
            if not hasattr(data, 'shape'):
                return True
            elif data.shape[0] == 0:
                return True
        return False
    
    def __dimensionsMatch(data0, data1):
        if data0.shape[0] != data1.shape[0]:
            return False
        return True

    def __shuffle(x, y):
        z = np.c_[x.reshape(len(x), -1), y.reshape(len(y), -1)]
        np.random.shuffle(z)
        x = z[:, :x.size//len(x)].reshape(x.shape)
        y = z[:, x.size//len(x):].reshape(y.shape)
        return x, y

    if __isEmpty(x, y):
        return None
    elif __dimensionsMatch(x, y) is False:
        return None

    p = int(x.shape[0] * proportion)

    x, y = __shuffle(x, y)

    return x[:p], x[p:], y[:p], y[p:]



