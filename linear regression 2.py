
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
ds=pd.read_csv('C:/Users/student/Desktop/Datasets/pptlr.csv')
y=ds.iloc[:,:-1].values
X=ds.iloc[:,:1].values

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=1/5, random_state=0)
regressor=LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.intercept_)
print(regressor.coef_)

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary VS Experience (Training Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary VS Experience (Test Set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()