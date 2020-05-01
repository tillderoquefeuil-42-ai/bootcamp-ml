import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from mylinearregression import MyLinearRegression as MyLR
from polynomial_model import add_polynomial_features

data = pd.read_csv("../../day01/resources/are_blue_pills_magics.csv")

X = np.array(data.Micrograms).reshape(-1,1)
Y = np.array(data.Score).reshape(-1,1)

x = []
myLR = []
for i in range(0, 9):
    print("For power {} :".format(i+2))
    x.append(add_polynomial_features(X, i+2))
    thetas = np.full((i+3, 1), 1.0)
    myLR.append(MyLR(thetas))

    alpha = 1 / math.pow(10, 3+i*2)
    myLR[i].fit_(x[i], Y, alpha=alpha, n_cycle=250000)

    MSE = myLR[i].mse_(x[i], Y)

    # print("thetas = {}".format(myLR[i].thetas))
    print("mse = {}\n".format(MSE))
    plt.bar(i+2, MSE, label="power {}".format(i+2))

plt.legend(prop={'size': 10})
plt.show()
