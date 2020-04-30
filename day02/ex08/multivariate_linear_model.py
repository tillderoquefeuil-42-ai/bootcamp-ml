# import numpy as np
import matplotlib.pyplot as plt


def plot_model(x, y, y_hat):
    
    plt.plot(x, y, 'co', x, y_hat, 'gx--')
    plt.show()
