# Module with python numeric computations
import numpy as numpy
# Modules used to make graphs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import random


# Creates variable t defined as np.array (0,...,10) in .1 intervals
x_1=np.random.random(1000)
# Creates variable u defined as np.array (-10, ..., 5) in some number of steps (in this case, the number of elements of t)
x_2=np.random.random(1000)
# This is a list comprehension which takes each pair l,v from the list [t,u] and maps l,v to the polynomial expression
# with some degree of randomness.
y=[ 50*l**2-24*l+5-58*v+(random.random()-.5)*l for l,v in np.array([x_1,x_2]).T]
# Converts y into a numpy array. The extra list '[' ']' symbols are added to turn y from a 1D array to a matrix. The extra '[' ']'
# symbols can be taken away by using '*y' which basically means empty out the contents of y.
y=np.array([y]).T


# Builds the data set in a matrix with first columns of 1's, second column x_1's, third columns x_1**2's, fourth column x_2
X=np.array([list(np.ones(len(x_1))),x_1,x_1**2,x_2]).T
# Solves the normal equation.
w=(np.linalg.inv(X.T.dot(X))).dot(X.T).dot(y)
#Calculates y_hat
y_hat=list(np.ones(len(x_1)))*w[0]+x_1*w[1]+x_1**2*w[2]+x_2*w[3]

# Creates helper lists V and L which can be sorted so that it can be plotted
# V is a matrix with columns sqrt(x_1^2+x_2^2) and y, that is consequently sorted wrt the first column. 
V=np.array([list(x_1**2+x_2**2),*y.T]).T
V=np.sort(V,0)
# Likewise, same for L.
L=np.array([list(x_1**2+x_2**2),y_hat]).T
L=np.sort(L,0)
# We plot V and L
plt.plot(V[:,0],V[:,1],color='b',label="Data Set")
plt.plot(L[:,0],L[:,1],color='g',label="Fitted Polynomial")
plt.legend()
# Calls matplotlib to actually draw the figure.
plt.show()

# Creates 'canvas' for 3D plot
fig = plt.figure()
# 3D plots are plotted to subplots of fig
ax = fig.add_subplot(111, projection='3d')
# creates a triangle between all nearest 3 points to make surface.
ax.plot_trisurf(x_1,x_2,y_hat)
ax.plot_trisurf(x_1,x_2,*y.T))
fig.tight_layout()
plt.show()



def polyfit(X,y,degree):
    """
    This function takes in a numpy array of m data points of n features
    with the corresponding 1D numpy array y values and determines
    a polynomial of degree n to fit it.
    """
    X_new=[list(np.ones(X.shape[0]))]
    for d in range(1,degree+1):
        for dim in range(X.shape[1]):
            X_new.append(X[:,dim]**d)
    X_new=np.array(X_new).T
    y=np.array([y]).T
    w=(np.linalg.inv(X_new.T.dot(X_new))).dot(X_new.T).dot(y)
    return w

