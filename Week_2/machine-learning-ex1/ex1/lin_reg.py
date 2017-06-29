#!/usr/bin/env python

import random
import Tkinter
import numpy as numpy
import scipy
import math
from numpy import random
from numpy import linalg
from scipy import integrate
from scipy import linalg
import pylab as P
import matplotlib.animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
#from showmat import showmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import *
from scipy.sparse.linalg import eigsh

"""The cost as a function of fitting parameters, theta. Returns a float."""
def J(x,y,theta):
    m= len(x)
    tot= 0
    for i in range(len(x)):
        tot+=(theta[0] + theta[1]*x[i] - y[i])**2
    return tot/(2*m)

"""The gradient of the cost function with respect to the fitting parameters, theta[0] and theta[1]. Returns
a 2x1 array"""
def dJ(x,y,theta):
    m= len(x)
    tot1= 0
    tot2= 0
    for i in range(len(x)):
        tot1+=(theta[0] + theta[1]*x[i] - y[i])
        tot2+=(theta[0] + theta[1]*x[i] - y[i])*x[i]
    return [tot1/m, tot2/m]

x= []
y= []

"""Opens text file of data and sorts columns into separate arrays"""
f=open("ex1data1.txt","r")
lines=f.readlines()
for l in lines:
    x.append(float(l.split(',')[0]))
    y.append(float(l.split(',')[1]))
f.close()

print len(x)

alpha= 0.02
theta= [1.0,1.0]
Niter= 1000
JvsStep= []
Step= []

"""Each iteration performs one step of gradient descent, simultaneously updating theta[0] and theta[1]"""
for n in range(Niter):
    JvsStep.append(J(x,y,theta))
    Step.append(n+1)
    gradJ= dJ(x,y,theta)
    tempTheta0= theta[0] - alpha*gradJ[0]
    tempTheta1= theta[1] - alpha*gradJ[1]
    theta[0]= tempTheta0
    theta[1]= tempTheta1

fity= []
for p in x:
    fity.append(theta[0]+theta[1]*p)


print theta

plt.plot(Step,JvsStep)
plt.ylabel("Cost Function J")
plt.xlabel("Number of Iterations of Gradient Descent")
plt.show()

plt.scatter(x,y)
plt.plot(x,fity)
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.show()
