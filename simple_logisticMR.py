from __future__ import division
import mrjob
import numpy as np
from mrjob.job import MRJob
from math import exp,log
import pickle


class spaceProtocol(object):

    
    def write(self, key, values):
        return ' '.join(str(v) for v in values)
       
              
class mrsimpleLogistic(MRJob):

     OUTPUT_PROTOCOL=spaceProtocol
        
     
     def configure_options(self):
         super(mrsimpleLogistic, self).configure_options()
         self.add_file_option('--theta')
         self.add_passthrough_option(
            '--numIters', type='int', help='Number of iterations')
            
     def get_theta(self):
        #f=open(self.options.theta)
        #theta=pickle.load(f)
        #f.close()
        theta=np.genfromtxt(self.options.theta)
        return theta
        
        
     def steps(self):
            return [self.mr(mapper=self.costDerivative,
                        reducer=self.sumCostDerivative)]
     def costDerivative(self,_,line):
        l=line.split()
        vector=np.array([float(i) for i in l])
        y=vector[0]
        data=vector[1:]
        if y!=1:
            y=0 
        theta=self.get_theta()
        x=np.append(1,data)
        thetaX=np.dot(theta,x)
        h=1/(1+exp(-thetaX))
        cost=-y*log(h)-(1-y)*log(1-h)
        dCost=(h-y)*x
        yield None,[cost,dCost.tolist()]
        
     def sumCostDerivative(self,key,values):
        theta=self.get_theta()
        cost=0
        dcost=np.array([0,0])
        cnt=0
        for i in values:
            cnt=cnt+1
            cost=cost+float(str(i[0]))
            dcost=dcost+np.array([float(k) for k in i[1]]) 
        alpha=5
        theta=theta-alpha*dcost/cnt 
        vals=np.append(cost,theta).tolist()
        yield None,vals
    
     
if __name__ == '__main__':
    mrsimpleLogistic.run()