from __future__ import with_statement
import mrjob
from mrjob.job import MRJob
from simple_logisticMR import mrsimpleLogistic
import os
import sys
import numpy as np
import ast
import pickle
THETA_FILE="/tmp/emr.theta"
if __name__ == '__main__':
    
    i=0
    while True:
            print "Iteration #%i" % i
            myLogistic=mrsimpleLogistic(args=['synData.txt']+['--theta='+THETA_FILE])
            with myLogistic.make_runner() as runner:
                runner.run()
                for line in runner.stream_output():
               
                   a=line.split('\n')
                   b=a[0]
                   c=b.split()
                   d=np.array([float(k) for k in c])
                   j=d[0]
                   theta=d[1:]
                   print j,theta
                   
                    
                    #j=float(c[0])
                    #theta=np.array(ast.literal_eval(c[1].strip()))
                    #print 'j=',j
                    #print '\ntheta=',theta
                   np.savetxt(THETA_FILE,theta,delimiter=',')
                    
            i+=1
            if i==50:
                break
                
            
    



       
