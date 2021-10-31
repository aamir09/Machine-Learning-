#Import the Libraries
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline


#Function to genertae Loss Function (y)
def gen_y(a):
    u=np.sin(a/10)*(5*(np.cos(a+10))-np.cos(a/2000))
    return u+20*np.sin(a)

#The derivative function
def der(a):
    u=(np.sin(a/10)*((np.sin(a/2000)/2000)-5*np.sin(a/10)))
    v=(np.cos(a/10)*(5*np.cos(a+10)- np.cos(a/200)))/10 
    return u + v + (20*np.cos(a))

def Gradient_Descent(W,W_prev=0,eta=0.004,tol=0.004,epochs=1):
    #Base Condition, from Equation 2
    if(W-W_prev<tol):
        print(f'Returning after {epochs} number of epochs')
        return W
    # We calculate the gradient value at W
    g=der(W) 
    # We memorize the the W 
    W_prev=W
    # We update the weight with the help of the previous one 
    # eta is the learning rate and tol is the tolerance
    W=W_prev-eta*g 
    #Itertaive Process and we also count the number of epochs 
    return Gradient_Descent(W,W_prev,epochs=epochs+1)

##################################  MAIN #######################################################

#Generate loss Function
x=np.linspace(15,25,500)
y=gen_y(x)

#Visulaize the plot
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(14,8))
ax[0].plot(x,y,color='r',linewidth=3)
ax[0].set_xlabel('Weight',color='g',fontsize=20)
ax[0].set_ylabel('Loss',color='r',fontsize=20)
ax[0].set_title('Loss Function',fontsize=25,color='b')

#Choose a random starting point and use the gradient descent function to reach the minima
start=15.25
best_weight=np.round(Gradient_Descent(start),2)
print(f'The optimal weight is {best_weight}')
#To plot iterations 
x_plot=np.linspace(start+0.1,best_weight-0.3,152)

##A Visual Plot of our excercise 
ax[1].plot(x,y,color='r',linewidth=3)
ax[1].scatter(best_weight,gen_y(best_weight),linewidth=16,label='Optimal Wight',color='blue')
ax[1].scatter(start,gen_y(start),linewidth=16,label='Start',color='orange')
ax[1].scatter(x_plot,gen_y(x_plot),linewidth=16,label='Iteration',color='#FDDF00',alpha=0.3)
ax[1].set_xlabel('Weight',color='g',fontsize=20)
ax[1].set_ylabel('Loss',color='r',fontsize=20)
ax[1].set_title('Finding Minima with Gradient Descent',fontsize=25,color='b')
ax[1].legend(markerscale=0.2)
plt.tight_layout()



