import numpy as np
import pandas as pd

from data_spliter import data_spliter
from my_logistic_regression import MyLogisticRegression as MyLR


data1 = pd.read_csv("../resources/solar_system_census.csv")
data2 = pd.read_csv("../resources/solar_system_census_planets.csv")

X = np.array(data1[['height', 'weight', 'bone_density']]).reshape(-1,3)
Y = np.array(data2.Origin).reshape(-1,1)

# print(X)
# print(Y)

zipcodes = np.array(data2.Origin.drop_duplicates())
zipcodes = np.sort(zipcodes)

x_train, x_test, y_train, y_test = data_spliter(X, Y, 0.8)

mylrs = []
for i in range(0, len(zipcodes)):
    print("For zipcode = {}".format(zipcodes[i]))
    y_train_z = np.array([[1 if y_train[j] == zipcodes[i] else 0 for j, x in enumerate(x_train)]]).T

    mylrs.append(MyLR([0., 0., 0., 0.]))

    cost = mylrs[i].cost_(x_train, y_train_z)
    print("Initial cost = {}".format(cost))

    mylrs[i].fit_(x_train, y_train_z, alpha=1e-4, n_cycle=100000)
    print("new thetas = {}".format(mylrs[i].thetas))

    cost = mylrs[i].cost_(x_train, y_train_z)
    print("Final cost = {}\n".format(cost))

# print(x_test.shape)
predictions = None
for i in range(0, len(mylrs)):
    predict = mylrs[i].predict_(x_test)
    if predictions is None:
        predictions = np.array(predict)
    else :
        predictions = np.append(predictions, predict, axis=1)

predictions = np.append(predictions, y_test, axis=1)
# print(predictions)

def maximumIndex(row):
    index = None
    value = None
    for i in range(0, len(row)):
        if index is None :
            index = i
            value = row[i]
        elif value < row[i]:
            index = i
            value = row[i]
    return index

good_predictions = 0
for i in range(0, predictions.shape[0]):
    row = predictions[i]
    y = int(row[4])
    index = maximumIndex(row[:4])
    if y == index:
        print('OK')
        good_predictions += 1
    print(row)

print("good predictions : {}%".format(good_predictions * 100 / predictions.shape[0]))