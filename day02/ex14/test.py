import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from data_spliter import data_spliter


from mylinearregression import MyLinearRegression as MyLR
from polynomial_model import add_polynomial_features

data = pd.read_csv("../../day01/resources/are_blue_pills_magics.csv")

X = np.array(data.Micrograms).reshape(-1,1)
Y = np.array(data.Score).reshape(-1,1)

x_train, x_test, y_train, y_test = data_spliter(X, Y, 0.8)

x = []
myLR = []
for i in range(0, 4):
    print("For power {} :".format(i+2))
    x.append(add_polynomial_features(x_train, i+2))
    thetas = np.full((i+3, 1), 1.0)
    myLR.append(MyLR(thetas))

    alpha = 1 / math.pow(10, (3+i*2))
    print("alpha = {}".format(alpha))

    myLR[i].fit_(x[i], y_train, alpha=alpha, n_cycle=250000)
    MSE = myLR[i].mse_(x[i], y_train)

    # print("thetas = {}".format(myLR[i].thetas))
    print("mse = {}\n".format(MSE))

    continuous_x = np.arange(1, 6.51, 0.01).reshape(-1,1) 
    x_ = add_polynomial_features(continuous_x, i+2)
    y_hat = myLR[i].predict_(x_)
    plt.scatter(x_test, y_test)
    plt.plot(continuous_x, y_hat, label="power {}".format(i+2))

plt.legend(prop={'size': 10})
plt.show()
