from __future__ import division
import mrjob
import numpy as np
from mrjob.job import MRJob
from math import exp,log
import pickle
import os


class spaceProtocol(object):

    
    def write(self, key, values):
        return ' '.join(str(v) for v in values)
       
              
class mrsimpleLogistic(MRJob):

     OUTPUT_PROTOCOL=spaceProtocol
        
     
     def configure_options(self):
         super(mrsimpleLogistic, self).configure_options()
         self.add_file_option('--theta')
         self.add_passthrough_option(
            '--labelValue', type='int', help='labelValue')
            
     def get_theta(self):
        #f=open(self.options.theta)
        #theta=pickle.load(f)
        #f.close()
        theta=np.genfromtxt(self.options.theta)
        return theta
        
        
     def steps(self):
            return [self.mr(mapper=self.costDerivative,
                        reducer=self.sumCostDerivative)]
     def costDerivative(self,_,inputFileName):
        f=open(inputFileName)
        cost=float(0)
        dcost=float(0)
        cnt=0
        for line in f:
            l=line.split()
            vector=np.array([float(i) for i in l])
            y=vector[0]
            data=vector[1:]
            if y!=self.options.labelValue:
                y=0 
            else: 
                y=1
            theta=self.get_theta()
            x=np.append(1,data)
            thetaX=np.dot(theta,x)
            h=1/(1+exp(-thetaX))
            cost=cost-y*log(h)-(1-y)*log(1-h)
            dcost=dcost+(h-y)*x
            cnt+=1
        f.close()
        yield None,[cost,dcost.tolist(),cnt]
        #yield None,l
        #yield None,inputFileName
        
     def sumCostDerivative(self,key,values):
        theta=self.get_theta()
        cost=0
        dcost=np.array([0,0])
        cnt=0
        for i in values:
            cost=cost+float(str(i[0]))
            dcost=dcost+np.array([float(k) for k in i[1]]) 
            cnt=cnt+float(str(i[2]))
        alpha=0.1
        theta=theta-alpha*dcost/cnt 
        vals=np.append(cost,theta).tolist()
        yield None,vals
        #yield None,
     
if __name__ == '__main__':
    mrsimpleLogistic.run()