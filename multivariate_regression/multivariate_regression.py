import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math

df= pd.read_csv('house.csv')

#handle missing values
median_bedroom=math.floor(df['bedrooms'].median())
df['bedrooms'].fillna(median_bedroom, inplace=True)

model = LinearRegression()

X = df[['area', 'bedrooms', 'age']]
y = df['price']

model.fit(X, y)

predict = model.predict([[3000, 3, 10]])  # Example prediction for a house with 3000 sqft, 3 bedrooms, and 10 years old

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

print("Prediction for a house with 3000 sqft, 3 bedrooms, and 10 years old:", predict[0])
