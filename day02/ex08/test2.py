import pandas as pd
import numpy as np

from mylinearregression import MyLinearRegression as MyLR
from multivariate_linear_model import plot_model

data = pd.read_csv("../resources/spacecraft_data.csv")

Xage = np.array(data[['Age']])
Xtp = np.array(data[['Thrust_power']])
Xtm = np.array(data[['Terameters']])

X = np.array(data[['Age', 'Thrust_power', 'Terameters']])
Y = np.array(data[['Sell_price']])

# exemple 2.a
print('exemple 2.a')

my_lreg = MyLR([1.0, 1.0, 1.0, 1.0])
print(my_lreg.mse_(X,Y))
# Output :
# 144044.877...

my_lreg.fit_(X,Y, alpha = 9.5e-5, n_cycle = 900000) # best predict
print(my_lreg.thetas)
# Output :
# array([[334.994...],[-22.535...],[5.857...],[-2.586...]])

print(my_lreg.mse_(X,Y))
# Output :
# 586.896999...


# plots
print('plots')

plot_model(Xage, Y, my_lreg.predict_(X))
plot_model(Xtp, Y, my_lreg.predict_(X))
plot_model(Xtm, Y, my_lreg.predict_(X))