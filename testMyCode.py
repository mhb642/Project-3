
import numpy as np
f=open('inputFileNames.txt')
i=0
data=[]
for line in f:
    print line
    line=line.split('\n')[0]
    data=data+np.genfromtxt(line).tolist()

data=np.array(data)
labels=data[:,0]
dataX=data[:,1]

thetaMatrix=np.genfromtxt('ThetaMatrix.txt',delimiter=',')
labelEst=np.zeros(np.size(labels))
for i in np.arange(0,400):
    labelEst[i]=np.argmax(np.dot(thetaMatrix,np.append(1,dataX[i])))
    
    