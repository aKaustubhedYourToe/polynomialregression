from numpy import array
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#original data in fahrenheit
temperatures = array([30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
ice_creams = array([200, 220, 260, 300, 320, 350, 400, 450, 500, 550])

#reshape the data for scikit-learn
X = temperatures.reshape(-1, 1)
y = ice_creams

#fit the linear regression model
model_linear = LinearRegression()
model_linear.fit(X, y)

#predictions
y_pred_linear = model_linear.predict(X)

#plotting the results
plt.scatter(temperatures, ice_creams, color='blue', label='Original data')
plt.plot(temperatures, y_pred_linear, color='green', linewidth=2, label='Linear regression')
plt.xlabel('Temperature (F)')
plt.ylabel('Ice Creams Sold')
plt.title('Ice Cream Sales vs Temperature')
plt.legend()
plt.grid(True)
plt.show()