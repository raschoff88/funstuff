# Module with python numeric computations
import numpy as numpy
# Module used to make graphs
import matplotlib.pyplot as plt
# Random module
import random

# Creates variable t defined as np.array (0,...,10) in .1 intervals
t=np.arange(0,10,.1)
# Creates variable u defined as np.array (-10, ..., 5) in some number of steps (in this case, the number of elements of t)
u=np.linspace(-10,5,len(t))
# This is a list comprehension which takes each pair l,v from the list [t,u] and maps l,v to the polynomial expression
# with some degree of randomness.
y=[ 2*l**2+4*l+5+3*v**2-4*v+2*(random.random()-.5)*l for l,v in np.array([t,u]).T]
# converts y into a numpy array
y=np.array([y]).T
# builds the data set in a matrix with first columns of 1's, second column t's, third columns u's, etc...
X=np.array([list(np.ones(len(t))),t,u,t**2,u**2]).T
# Solves the normal equation.
w=(np.linalg.inv(X.Tdot(X))).dot(X.T).dot(y)
