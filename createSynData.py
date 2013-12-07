#this script generates the sample input for my MapReduce Logistic Regression
# the data comes from two gaussian distributions with two variances.
# D1 ~ N(15,1)         label 0
# D2 ~ N(-15,1)        label 1
# it is saved in a text file with format as follows:
# label    D
#  1      14.9 

import numpy as np
from numpy import random
N=100
# Generate D1:
mu1,sigma1=-15,1   #mean and standard deviation
D1=random.normal(mu1,sigma1,N)

mu2,sigma2=0,1
D2=random.normal(mu2,sigma2,N)

dataX=np.append(D1,D2)
dataY=np.append(0*np.ones([N,1]),1*np.ones([N,1])).T
data=np.zeros([2*N,2])
data[:,0]=dataY
data[:,1]=dataX
np.savetxt('synData1.txt',data,delimiter=' ')