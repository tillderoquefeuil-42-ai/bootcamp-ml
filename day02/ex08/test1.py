import pandas as pd
import numpy as np

from mylinearregression import MyLinearRegression as MyLR
from multivariate_linear_model import plot_model

data = pd.read_csv("../resources/spacecraft_data.csv")

Xage = np.array(data[['Age']])
Xtp = np.array(data[['Thrust_power']])
Xtm = np.array(data[['Terameters']])

Y = np.array(data[['Sell_price']])

# exemple 1.a
print('exemple 1.a')

myLR_age = MyLR([[650.0], [-10.0]])

myLR_age.fit_(Xage[:,0].reshape(-1,1), Y, alpha = 2.5e-3, n_cycle = 50000)
print(myLR_age.thetas)

MSE_age = myLR_age.mse_(Xage[:,0].reshape(-1,1),Y) 
print(MSE_age)


# exemple 1.b
print('exemple 1.b')

myLR_thrust = MyLR([[35.0], [4.0]])

myLR_thrust.fit_(Xtp[:,0].reshape(-1,1), Y, alpha = 1e-4, n_cycle = 100000)
print(myLR_thrust.thetas)

MSE_thrust = myLR_thrust.mse_(Xtp[:,0].reshape(-1,1),Y) 
print(MSE_thrust)


# exemple 1.c
print('exemple 1.c')

myLR_distance = MyLR([[715.0], [-2.0]])

myLR_distance.fit_(Xtm[:,0].reshape(-1,1), Y, alpha = 2.5e-5, n_cycle = 50000)
print(myLR_distance.thetas)

MSE_distance = myLR_distance.mse_(Xtm[:,0].reshape(-1,1),Y) 
print(MSE_distance)


# plots
print('plots')

plot_model(Xage, Y, myLR_age.predict_(Xage))
plot_model(Xtp, Y, myLR_thrust.predict_(Xtp))
plot_model(Xtm, Y, myLR_distance.predict_(Xtm))