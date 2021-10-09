#Import the Libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Let's create a data frame and have a look at it
X=np.linspace(0,1,100)
Y=3*X+2
Y=Y+np.random.normal(0,1,Y.shape)
X = X.reshape(-1, 1)
plt.scatter(X,Y)
plt.xlabel('X',fontsize=15)
plt.ylabel('Y',fontsize=15)
plt.title('Relationship b/w X and Y',fontsize=17)

#let's select the number of bootstraps we want
boot=500
#Create a bootstrap
df=pd.DataFrame(X,columns=['X'])
df['Y']=Y

#Iteration Begins 
X = X.reshape(-1, 1)
betas=[]
preds=[]
for i in range(boot):
    df1=df.sample(len(df),replace=True) #random sample with replacement, frac=1 mean size will be the size of data set
    lr=LinearRegression()
    model=lr.fit(df1[['X']],df1['Y'])
    y_pred=model.predict(X)
    plt.plot(X,y_pred,color='red',alpha=0.1)
    betas.append(model.coef_[0])
    preds.append(y_pred)
plt.scatter(X,Y,label='Original')
plt.legend()
plt.title('Bootstraping Estiamtions',fontsize=15,color='green')


#How about wathching the distributions
fig,ax=plt.subplots(figsize=(5,5))
plt.hist(betas,label='Beta Frequency')
plt.xlabel('Beta Values',fontsize=10)
plt.ylabel('Frequencies',fontsize=10)
plt.title('Beta Distribution',fontsize=12,color='red')
plt.axvline(np.array(betas).mean(axis=0),color='red',label='Mean')
plt.legend()
plt.tight_layout()

#Calculating 95% Confidence Intervals for beta1
CI=(np.percentile(betas,2.5),np.percentile(betas,97.25))
print(CI)


## Distribution with CI Visualization
fig,ax=plt.subplots(figsize=(6,6))
plt.hist(betas,label='Beta Frequency',alpha=0.5)
plt.xlabel('Beta Values',fontsize=15)
plt.ylabel('Frequencies',fontsize=15)
plt.title('Beta Distribution',fontsize=17,color='red')
plt.axvline(np.array(betas).mean(axis=0),color='red',label='Mean')
plt.axvline(CI[0],color='green',label='95% Confidence Interval')
plt.axvline(CI[1],color='green',label='95% Confidence Interval')
plt.legend(fontsize='small')
plt.tight_layout()

#95% Confidence Interval of precdiction
fig,ax=plt.subplots(figsize=(6,6))
mean=np.array(preds).mean(axis=0)
plt.plot(X,mean,label='Mean at each bootstrap',color='blue')
plt.xlabel('Y-values',fontsize=15)
plt.ylabel('X-values',fontsize=15)
plt.title('95% Confidence Interval of Predictions',fontsize=17,color='red')
plt.fill_between(X.flatten(),mean-(2*np.std(np.array(preds),axis=0)),mean+(2*np.std(np.array(preds),axis=0)),alpha=0.2)
plt.legend(fontsize='small')
plt.scatter(X,Y)
plt.tight_layout()