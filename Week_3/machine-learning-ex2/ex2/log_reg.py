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

"""The logistic function. Returns a float between 0 and 1."""
def sigmoid(z):
    temp= 1.0 + numpy.exp(-z)
    return 1.0/temp

"""Returns the cost as a function of fitting parameters, theta, given training data x and y."""
def J(x,y,theta):
    m= len(x)
    n= len(x[0])
    tot= 0.0
    for i in range(m):
        z= 0.0
        for j in range(n):
            z+=theta[j]*x[i][j]
        if y[i]:
            tot+= -numpy.log(sigmoid(z))
        else:
            #print sigmoid(z)
            tot+= -numpy.log(1.0-sigmoid(z))
    return tot/m

"""Returns the gradient of J(theta) as a list"""
def dJ(x,y,theta):
    m= len(x)
    n= len(x[0])
    grad= []
    for j in range(n):
        tot= 0
        for i in range(m):
            z= 0.0
            for l in range(n):
                z+=theta[l]*x[i][l]
            tot+=(sigmoid(z)-float(y[i]))*x[i][j]
        grad.append(tot/m)
    return grad

"""Minimizes cost function J with gradient dJ as a function of thetaInit, given training data x and y.
Returns fitting parameters, theta, and costs at each iteration."""
def gradDescent(dJ,J,x,y,thetaInit,N,alpha):
    theta= thetaInit[:]
    thetaTemp= thetaInit[:]
    nFeat= len(x[0])
    costs= []
    for i in range(N):
        costs.append(J(x,y,theta))
        gradTemp= dJ(x,y,theta)
        for j in range(nFeat):
            
            thetaTemp[j]=theta[j] - alpha*gradTemp[j]
        theta= thetaTemp[:]
    return [theta,costs]

xtest=[]
ytest=[]

"""Opens text file of data and sorts columns into separate arrays"""
f=open("ex2data1.txt","r")
lines=f.readlines()
for l in lines:
    xtest.append([1.0] + [float(i)/100.0 for i in l.split(',')[:-1]] )
    ytest.append(int(l.split(',')[-1]))
f.close()

"""Set initial paramters and number of iterations for gradient descent"""
theta0= [0.0 for i in range(len(xtest[0]))]
Niter= 5000

"""Minimize cost function via gradient descent to obtain fitting parameters, theta1,
and costs as a function of iteration, Js"""
temp= gradDescent(dJ,J,xtest,ytest,theta0,Niter, 2.5)
theta1= temp[0]
Js= temp[1]

zeros= [[],[]]
ones= [[],[]]
for i in range(len(xtest)):
    if ytest[i]:
        ones[0].append(xtest[i][1])
        ones[1].append(xtest[i][2])
    else:
        zeros[0].append(xtest[i][1])
        zeros[1].append(xtest[i][2])

"""Plot costs as a function of iteration"""
plt.plot(Js)
plt.show()

dom= numpy.linspace(0,1.0,50)
ran= -(theta1[1]/theta1[2])*dom - (theta1[0]/theta1[2])

"""Plot training data and decision boundary"""
plt.plot(dom, ran, c="green")
plt.scatter(ones[0],ones[1],c="red",s=100.0)
plt.scatter(zeros[0],zeros[1],s=100.0)
plt.show()


