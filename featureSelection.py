# Import all the required libraries 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,Normalizer, MinMaxScaler
import matplotlib.pyplot as plt
# %matplotlib inline 
#Uncomment the above line only if you are using jupyter notebooks

#To improve visulaization
plt.rcParams['figure.figsize']=(10,20)
plt.rcParams['axes.facecolor'] = 'black'

# Generate a uniform distribution and Visulaize it with matplotlib
x=np.random.uniform(low=5,high=10,size=10000)

#Visualization of uniform distribution using Matplotlib
fig,(ax1,ax2,ax3,ax4)=plt.subplots(nrows=4,ncols=1)
ax1.hist(x,color='#C3073F',width=0.5,alpha=0.9)
ax1.set_title('Uniform Distribution of Feature x',fontsize=19,color='#C3073F')
ax1.grid(True,alpha=0.4)
ax1.set_xlabel('Range',fontsize=17)
ax1.set_ylabel('Frequency',fontsize=17)


#Using Standardization on x
ss=StandardScaler()
ss_x=ss.fit_transform(x.reshape(-1,1)) # Standard Scaler expects a 2D array of features 
ax2.hist(ss_x,color='#C3073F',width=0.5,alpha=0.9)
ax2.set_title('Standard Uniform Distribution of Feature x',fontsize=19,color='#C3073F')
ax2.grid(True,alpha=0.4)
ax2.set_xlabel('Range',fontsize=17)
ax2.set_ylabel('Frequency',fontsize=17)

#Using Min-Max Scaling
mms=MinMaxScaler()
mms_x=mms.fit_transform(x.reshape(-1,1))
ax3.hist(mms_x,color='#C3073F',width=0.5,alpha=0.9)
ax3.set_title('MinMax Scaled Uniform Distribution of Feature x',fontsize=19,color='#C3073F')
ax3.grid(True,alpha=0.4)
ax3.set_xlabel('Range',fontsize=17)
ax3.set_ylabel('Frequency',fontsize=17)

#l2 Normalization
l2_norm=Normalizer(norm='l2')
l2_norm_x=l2_norm.fit_transform(x.reshape(-1,1))
ax4.hist(l2_norm_x,color='#C3073F',width=0.5,alpha=0.9)
ax4.set_title('l2 Scaled Uniform Distribution of Feature x',fontsize=19,color='#C3073F')
ax4.grid(True,alpha=0.4)
ax4.set_xlabel('Range',fontsize=17)
ax4.set_ylabel('Frequency',fontsize=17)



plt.tight_layout()