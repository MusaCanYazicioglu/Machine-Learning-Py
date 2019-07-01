# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Siplitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_Y = StandardScaler()
Y_train = sc_Y.fit_transform(Y_train)"""

# Fitting Linear Regression to the Dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

# Fitting Polinomial Regression on the Dataset (Degree=2)
from sklearn.preprocessing import PolynomialFeatures
poly_reg2 = PolynomialFeatures(degree = 2)
X_poly2 = poly_reg2.fit_transform(X)
poly_reg2.fit(X_poly2, Y)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly2, Y)

# Fitting Polinomial Regression on the Dataset (Degree=3)
from sklearn.preprocessing import PolynomialFeatures
poly_reg3 = PolynomialFeatures(degree = 3)
X_poly3 = poly_reg3.fit_transform(X)
poly_reg3.fit(X_poly3, Y)
lin_reg3 = LinearRegression()
lin_reg3.fit(X_poly3,Y)

# Fitting Polinomial Regression on the Dataset (Degree=4)
from sklearn.preprocessing import PolynomialFeatures
poly_reg4 = PolynomialFeatures(degree = 4)
X_poly4 = poly_reg4.fit_transform(X)
poly_reg4.fit(X_poly4, Y)
lin_reg4 = LinearRegression()
lin_reg4.fit(X_poly4,Y)

# Visualising the Linear Regression Results
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression Results (Degree=2)
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg2.predict(poly_reg2.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression Results (Degree=3)
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg3.predict(poly_reg3.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression Results (Degree=4)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, lin_reg4.predict(poly_reg4.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a New Result With Linear Regression
lin_reg.predict([[6.5]])

# Predicting a New Result With Polynomial Regression
lin_reg4.predict(poly_reg4.fit_transform([[6.5]]))