import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ridge import MyRidge as MR
from mylinearregression import MyLinearRegression as MLR

from data_spliter import data_spliter
from polynomial_model import add_polynomial_features

data = pd.read_csv("../../day02/resources/spacecraft_data.csv")

X = np.array(data[['Age', 'Thrust_power', 'Terameters']]).reshape(-1,3)
Y = np.array(data[['Sell_price']])

# Data Splitting
print("Split data (training/test set)\n")
x_train, x_test, y_train, y_test = data_spliter(X, Y, 0.8)


# Training
print("Train models")

models = []
# thetas = [0., 0., 0., 0., 0.]
thetas = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
alpha = 1e-13
n_cycle = 50000
lambda_ = 0.

x_train = add_polynomial_features(x_train, 3)
x_test = add_polynomial_features(x_test, 3)

print("\tLinear Regression model (x1)")
m0 = MLR(thetas=thetas, alpha=alpha, n_cycle=n_cycle)
print(m0.mse_(x_train, y_train))
m0.fit_(x_train, y_train)
print(m0.mse_(x_train, y_train))

models.append(m0)

print("\tRidge Regression models (x9)")
for i in range(0, 9):
    lambda_ += 0.1
    mi = MR(thetas=thetas, alpha=alpha, n_cycle=n_cycle, lambda_=lambda_)
    print(mi.mse_(x_train, y_train))
    mi.fit_(x_train, y_train)
    print(mi.mse_(x_train, y_train), "\n")
    models.append(mi)


# Plots
print("Plots models")


length = len(models)
plt.scatter(x_test[:,0], y_test)

for i in range(0, length):
    l = i * 0.1

    y_hat = models[i].predict_(x_test)
    plt.plot(x_test[:,0], y_hat, label="lambda_ = {:.1f}".format(l))

plt.legend(prop={'size': 10})
plt.show()
