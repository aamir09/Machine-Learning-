#Import the necessary Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
%matplotlib inline

#Read The Data Set 
df=pd.read_csv('Advertising.csv')
#View your data 
df.head(5)
# print(len(df))

#Plot a scatter plot to view the relationship between x and y
x=df[['TV']] # predictor 
y=df['Sales'] #response or output variable
plt.scatter(x,y)
plt.xlabel('TV budget in 1000$' ,fontsize=20)
plt.ylabel('Sales in 1000$',fontsize=20)
plt.title('Scatter Plot', fontsize=25)

#Split the Data 
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7)

#Create elbow plot 
fig,ax=plt.subplots(figsize=(10,10))
k_list=np.arange(1,80,1)
knn_dict={}
minimum=100000000000000000
for i in k_list:
    knn=KNeighborsRegressor(n_neighbors=int(i))
    model_knn=knn.fit(x_train,y_train)
    y_knn_pred=model_knn.predict(x_test)
    mse=mean_squared_error(y_test,y_knn_pred)
    knn_dict[i]=mse
ax.plot(knn_dict.keys(),knn_dict.values())
ax.set_xlabel('K-VALUE', fontsize=20)
ax.set_ylabel('MSE' ,fontsize=20)
ax.set_title('ELBOW PLOT' ,fontsize=28)

### Create Model with the optimal nunmber of k
knn=KNeighborsRegressor(n_neighbors=9)
model_knn=knn.fit(x_train,y_train)
y_knn_pred=model_knn.predict(x_test)

## Calculate the performance metrics 
print(mean_squared_error(y_test,y_knn_pred))
print(r2_score(y_test,y_knn_pred))
