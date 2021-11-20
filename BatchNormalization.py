import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
%matplotlib inline




#Generating our function that we will estimate using a neural network
x=np.linspace(0.5,2,200)
y=np.sin(x)*np.cos(5*x)+10

#Adding white Gaussian Noise to y
Y=y+np.random.normal(loc=0,scale=0.3,size=len(x))




fig,ax1=plt.subplots(figsize=(10,10))
ax1.plot(x,y,label='Original Data',color='blue',linewidth=3)
ax1.scatter(x,Y,label='Original Data with White Noise',color='r')
ax1.set_title('Data Generation',fontsize=18)
ax1.set_xlabel(r'$X$',fontsize=16)
ax1.set_ylabel(r'$Y$',fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.tick_params(axis='both', which='minor', labelsize=14)
ax1.legend(borderpad=1.5,fontsize=11)
plt.show()




# Splitting Data into test and train
x_train,x_val,y_train,y_val=train_test_split(x,Y,train_size=0.75)




#Plot the Training and Validation Data 
fig,ax2=plt.subplots(figsize=(10,10))
ax2.plot(x,y,label='Original Data',color='#A0E806',linewidth=4)
ax2.scatter(x_train,y_train,label='Train Data',color='#5A5E5B',linewidths=1.5)
ax2.scatter(x_val,y_val,label='validation Data',color='#FFC000',linewidths=1.5)
ax2.set_title('Visualizing Training and Validation Data ',fontsize=18)
ax2.set_xlabel(r'$X$',fontsize=16)
ax2.set_ylabel(r'$Y$',fontsize=16)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='minor', labelsize=14)
ax2.legend(borderpad=1.5,fontsize=14)
plt.show()




# Building Neural Network Without Normalization
inputs=keras.Input(shape=1)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(inputs)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(x)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(x)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(x)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(x)
outputs=keras.layers.Dense(1,activation='linear')(x)

#Model Creation

model=keras.Model(inputs,outputs)
model.summary()

#Compiling our Model 
model.compile(loss='mse',metrics='mse',optimizer=tf.optimizers.Adam(learning_rate=0.0001,))

#Trainig our model

history=model.fit(x_train,y_train,epochs=500,validation_split=0.2,batch_size=32,verbose=0)

#Predictions and Mean Square Error Calculation

preds=model.predict(x_val)
mse=mean_squared_error(y_val,preds)
print(f'The Mean Squared Error Without Normalization is: {np.round(mse,3)}')






#Plot the Loss Function
fig,ax3=plt.subplots(figsize=(10,10))
ax3.plot(history.history['loss'],label='Training Loss',linewidth=4)
ax3.plot(history.history['val_loss'],label='Validation Loss',linewidth=4)
ax3.set_title('Convergence of Loss in  Neural Network',fontsize=18)
ax3.set_xlabel(r'$Epochs$',fontsize=16)
ax3.set_ylabel(r'$Loss$',fontsize=16)
ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.tick_params(axis='both', which='minor', labelsize=14)
ax3.legend(borderpad=1.5,fontsize=14)
plt.show()





# Building Neural Network With Batch Normalization
inputs=keras.Input(shape=1)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(inputs)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(x)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(x)
x=keras.layers.Dense(100,kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(activation='relu')(x)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(x)
outputs=keras.layers.Dense(1,activation='linear')(x)

#Model Creation

model1=keras.Model(inputs,outputs)
model1.summary()

#Compiling our Model 
model1.compile(loss='mse',metrics='mse',optimizer=tf.optimizers.Adam(learning_rate=0.08,))

#Trainig our model

history1=model1.fit(x_train,y_train,epochs=500,validation_split=0.2,batch_size=32,verbose=0)

#Predictions and Mean Square Error Calculation

preds=model1.predict(x_val)
mse=mean_squared_error(y_val,preds)
print(f'The Mean Squared Error With Normalization is: {np.round(mse,3)}')




#Plot the Loss Function
fig,ax4=plt.subplots(figsize=(10,10))
ax4.plot(history1.history['loss'],label='Training Loss',linewidth=4)
ax4.plot(history1.history['val_loss'],label='Validation Loss',linewidth=4)
ax4.set_title('Convergence of Loss in  Neural Network',fontsize=18)
ax4.set_xlabel(r'$Epochs$',fontsize=16)
ax4.set_ylabel(r'$Loss$',fontsize=16)
ax4.tick_params(axis='both', which='major', labelsize=14)
ax4.tick_params(axis='both', which='minor', labelsize=14)
ax4.legend(borderpad=1.5,fontsize=14)
plt.show()


