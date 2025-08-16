import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('homeprices.csv')

# Prepare the data
X = df[['area']]  # Feature(s) as a 2D array
y = df['price']   # Target variable

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Make a prediction
predicted_price = model.predict([[3300]])
print(f"Predicted price for area 3300: {predicted_price[0]}")

# Print model parameters
print(f"Coefficient: {model.coef_[0]}")

# Plotting
import matplotlib.pyplot as plt

# Scatter plot of the data
plt.scatter(X, y, color='blue', label='Actual data')

# Regression line
plt.plot(X, model.predict(X), color='red', label='Regression line')

plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Linear Regression: Area vs Price')
plt.legend()
plt.show()