from __future__ import with_statement
import mrjob
from mrjob.job import MRJob
from batch_logisticMR import mrsimpleLogistic
import os
import sys
import numpy as np
from numpy import random
import ast
import pickle

THETA_FILE="/tmp/emr.theta"
if __name__ == '__main__':
    numLabels=3
    numFeatures=1    
    thetaMatrix=np.zeros([numLabels,numFeatures+1])   
    for l in np.arange(0,numLabels):
        theta=random.uniform(0,1,2)
        np.savetxt(THETA_FILE,theta,delimiter=',')
        i=0
        while True:
                print "Iteration #%i" % i
                
                myLogistic=mrsimpleLogistic(args=['inputFileNames.txt']+['--theta='+THETA_FILE]+['--labelValue='+str(l)])
                with myLogistic.make_runner() as runner:
                    runner.run()
                    for line in runner.stream_output():
                        #print line
                        a=line.split('\n')
                        b=a[0]
                        c=b.split()
                        d=np.array([float(k) for k in c])
                        j=d[0]
                        theta=d[1:]
                        print j,theta
                        #print theta
                        np.savetxt(THETA_FILE,theta,delimiter=',')
                        
                i+=1
                if i==1000:
                    break
        thetaMatrix[l,:]=theta

    np.savetxt('ThetaMatrix.txt',thetaMatrix,delimiter=',')
            
            
    



       
