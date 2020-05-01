import matplotlib.pyplot as plt


def plot_model(x, y, y_hat):
    
    plt.plot(x, y, 'bo', x, y_hat, 'c.')
    plt.grid(True)
    plt.show()
