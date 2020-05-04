import numpy as np

def reg_log_cost_(y, y_hat, theta, lambda_):
    
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

    if __isEmpty(y_hat, y):
        return None
    elif __dimensionsMatch(y_hat, y) is False:
        return None

    t = theta[:]
    t[0] = 0

    m = y.shape[0]
    ones = np.ones((m,1))

    result = (-1/m) * np.sum( y * np.log(y_hat) + (ones - y) * np.log(ones - y_hat) ) + ( lambda_/(2 * m) * t.dot(t) )
    return result
