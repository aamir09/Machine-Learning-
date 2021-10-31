#Import the necessary Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
#%matplotlib inline

#Read The Data Set 
df=pd.read_csv('Salary_Data.csv')
#View your data 
df.head(5)

#Draw a scatter Plot to see relationship between the variables
x=df[['YearsExperience']] # predictor 
y=df['Salary'] #response or output variable
plt.scatter(x,y)
plt.xlabel('Years of Experience' ,fontsize=20)
plt.ylabel('Salary',fontsize=20)
plt.title('Scatter Plot')

#Split Data set 
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7)

#Model Developement 
lr=LinearRegression()
model=lr.fit(x_train,y_train)
print('The Coefficients for this model is ',model.coef_)
y_train_pred=model.predict(x_train)
mse_train=mean_squared_error(y_train,y_train_pred)
print(f"MSE for Train Data set is {mse_train}")
y_test_pred=model.predict(x_test)
mse_test=mean_squared_error(y_test,y_test_pred)
print(f"MSE for Train Data set is {mse_test}")

#Plot your regression result
fig,ax=plt.subplots(figsize=(8,8))
ax.scatter(x,y,label='Original Data')
plt.plot(x_test,y_test_pred,color='red',label='Predictions')
plt.xlabel('Years of Experience' ,fontsize=20)
plt.ylabel('Salary',fontsize=20)
plt.title('Simple Linear Regression',fontsize=22)
ax.legend()

