import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math
from sklearn.model_selection import train_test_split

df= pd.read_csv('house.csv')

#handle missing values
median_bedroom=math.floor(df['bedrooms'].median())
df['bedrooms'].fillna(median_bedroom, inplace=True)

model = LinearRegression()

X= df[['area','bedrooms','age']]
y=df['price']

#split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=20)

#train the model
model.fit(x_train,y_train) 

#test the model
predict = model.predict(x_test) 

print(predict)

print(y_test.values)

print(model.score(x_test, y_test))



import pickle
from sklearn.metrics import r2_score

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)


