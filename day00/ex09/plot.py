import matplotlib.pyplot as plt

def plot_with_cost(x, y, theta):
    """
    Plot the data and prediction line from three non-empty numpy.ndarray. 
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        y: has to be an numpy.ndarray, a vector of dimension m * 1. 
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns: 
        Nothing.
    Raises:
        This function should not raise any Exception.
    """

    if x.shape[0] == 0 or y.shape[0] == 0 or theta.shape[0] == 0:
        return None

    result = []
    for i in x:
        value = theta[0] + theta[1] * i
        result.append(value)

    plt.plot(x, y, 'bo', x, result, 'y-')

    for i in range(0, len(x)):
        plt.plot([x[i], x[i]], [y[i], result[i]], 'r--')

    plt.show()
