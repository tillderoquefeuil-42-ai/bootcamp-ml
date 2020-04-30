import pandas as pd
import numpy as np

from mylinearregression import MyLinearRegression as MyLR
from multivariate_linear_model import plot_model

data = pd.read_csv("../resources/spacecraft_data.csv")

X = np.array(data[['Age']])
Y = np.array(data[['Sell_price']])

myLR_age = MyLR([[1000.0], [-1.0]])

myLR_age.fit_(X[:,0].reshape(-1,1), Y, alpha = 2.5e-5, n_cycle = 50000)

RMSE_age = myLR_age.mse_(X[:,0].reshape(-1,1),Y) 
print(RMSE_age)
# Output :
# 57636.77729...

plot_model(X, Y, myLR_age.predict_(X))