import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from my_logistic_regression import MyLogisticRegression as MyLR
from data_spliter import data_spliter
from polynomial_model import add_polynomial_features
from other_metrics import f1_score_


data1 = pd.read_csv("../../day03/resources/solar_system_census.csv")
data2 = pd.read_csv("../../day03/resources/solar_system_census_planets.csv")

X = np.array(data1[['height', 'weight', 'bone_density']]).reshape(-1,3)
Y = np.array(data2.Origin).reshape(-1,1)

zipcodes = np.array(data2.Origin.drop_duplicates())
zipcodes = np.sort(zipcodes)

# Data Splitting
print("Split data (training/test set)\n")

x_train, x_test, y_train, y_test = data_spliter(X, Y, 0.8)

x_train = add_polynomial_features(x_train, 3)
x_test = add_polynomial_features(x_test, 3)

# Training
print("Train models")

thetas = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
alpha = 1e-4
n_cycle=100000
lambda_ = 0

models = []
for i in range(0, len(zipcodes)):
    lambda_ += 0.1

    print("For zipcode = {}".format(zipcodes[i]))
    y_train_z = np.array([[1 if y_train[j] == zipcodes[i] else 0 for j, x in enumerate(x_train)]]).T

    mn = MyLR(thetas=thetas, alpha=alpha, n_cycle=n_cycle, penalty='l2', lambda_=lambda_)
    mn.fit_(x_train, y_train_z)
    print("new thetas = {}".format(mn.thetas))

    models.append(mn)


for i in range(0, len(models)):
    l = i * 0.1

    y_hat = models[i].predict_(x_test)
    print(f1_score_(y_test, y_hat))

