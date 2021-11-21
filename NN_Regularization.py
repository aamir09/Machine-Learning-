import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
# %matplotlib inline

#Loading our data set 
X=pd.DataFrame(load_iris()['data'],columns=load_iris()['feature_names'])
y=load_iris()['target'].reshape(-1,1)
#View Dataset
X.head(5)

#Encoding our target Variable
encoder=OneHotEncoder()
y=encoder.fit_transform(y)
y=y.toarray()
#View the shape our new target variable
print(f'The shape of response is {y.shape}')

#Splitting Data into train and valdiation data 
x_train,x_val,y_train,y_val=train_test_split(X,y,train_size=0.75,stratify=y)

#Creating THree Different Models i.e; Standard MLP, MLP with dropout, MLP with Norm Regualrization

#Standard MLP
inputs=keras.Input(shape=4)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(inputs)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(x)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(x)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(x)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(x)
outputs=keras.layers.Dense(3,activation='softmax')(x)

model=keras.Model(inputs,outputs)
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=keras.optimizers.Adam(learning_rate=0.001))
history=model.fit(x_train,y_train,epochs=300,validation_split=0.2,batch_size=64,verbose=0)

#MLP with Dropout 
inputs=keras.Input(shape=4)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(inputs)
# x=keras.layers.Dropout(0.5)(x)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(x)
x=keras.layers.Dropout(0.5)(x)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(x)
x=keras.layers.Dropout(0.5)(x)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(x)
# x=keras.layers.Dropout(0.5)(x)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(x)
# x=keras.layers.Dropout(0.5)(x)
outputs=keras.layers.Dense(3,activation='softmax')(x)

model_drop=keras.Model(inputs,outputs)
model_drop.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=keras.optimizers.Adam(learning_rate=0.001))
history_drop=model_drop.fit(x_train,y_train,epochs=300,validation_split=0.2,batch_size=64,verbose=0)

#MLP with Norm Regularization

inputs=keras.Input(shape=4)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None),kernel_regularizer=keras.regularizers.l2(1e1))(inputs)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None),kernel_regularizer=keras.regularizers.l2(1e1))(x)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(x)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(x)
x=keras.layers.Dense(100,activation='relu',kernel_initializer=tf.keras.initializers.GlorotNormal(seed=None))(x)
outputs=keras.layers.Dense(3,activation='softmax')(x)

model_norm=keras.Model(inputs,outputs)
model_norm.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=keras.optimizers.Adam(learning_rate=0.001))
history_norm=model_norm.fit(x_train,y_train,epochs=300,validation_split=0.2,batch_size=64,verbose=0)


#Plot the Loss Function
fig,(ax1,ax2,ax3)=plt.subplots(nrows=1,ncols=3,figsize=(60,15))
axes=[ax1,ax2,ax3]
histories=[history,history_drop,history_norm]
for i,j in zip(axes,histories):
    i.plot(j.history['loss'],label='Training Loss',linewidth=4)
    i.plot(j.history['val_loss'],label='Validation Loss',linewidth=4)
    i.set_title('Convergence of Loss in  Neural Network',fontsize=35)
    i.set_xlabel(r'$Epochs$',fontsize=30)
    i.set_ylabel(r'$Loss$',fontsize=30)
    i.tick_params(axis='both', which='major', labelsize=30)
    i.tick_params(axis='both', which='minor', labelsize=30)
    i.legend(borderpad=1.5,fontsize=40)
plt.show()

#Plot the Loss Function
fig,ax4=plt.subplots(figsize=(15,10))
ax4.plot(history.history['val_accuracy'],label='Standard MLP',linewidth=4)
ax4.plot(history_drop.history['val_accuracy'],label='MLP with Dropout',linewidth=4)
ax4.plot(history_norm.history['val_accuracy'],label='MLP with Norm Regularization',linewidth=4)
ax4.set_title('Validation Accuracies in  Neural Network',fontsize=18)
ax4.set_xlabel(r'$Epochs$',fontsize=16)
ax4.set_ylabel(r'$Loss$',fontsize=16)
ax4.tick_params(axis='both', which='major', labelsize=14)
ax4.tick_params(axis='both', which='minor', labelsize=14)
ax4.legend(borderpad=1.5,fontsize=14)
plt.show()
