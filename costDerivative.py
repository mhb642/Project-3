from __future__ import division

import numpy as np
from math import exp,log
import pickle
from mrjob.job import MRJob
from itertools import combinations

class SemicolonValueProtocol(object):

    

    def write(self, key, values):
        return ';'.join(str(v) for v in values)

class MRCostAndDerivative(MRJob):
    
     OUTPUT_PROTOCOL = SemicolonValueProtocol
     def __init__(self, args):
        MRJob.__init__(self, args)  
        
     def steps(self):
        return [self.mr(mapper=self.CostDerivative,
                        reducer=self.SumCostDerivative)]
      
     def configure_options(self):
         super(MRCostAndDerivative, self).configure_options()
         self.add_file_option('--theta')
         self.add_passthrough_option(
            '--labelVal', type='int', help='Number of clusters')
         
     def get_theta(self):
         f=open(self.options.theta)
         theta=pickle.load(f)
         f.close()
         return theta
     
    
     def CostDerivative(self,_,line):
        l=line.split()
        y=int(l[0])
        label=self.options.labelVal
        if y!=label:
            y=0
        else:
            y=1 
        x=np.array([int(i) for i in l[1:]])
        x=np.append(1,x)
        
        theta=self.get_theta()
        thetaArray=np.array([float(i) for i in theta])
        thetaX=np.dot(thetaArray,x)
        interValue1=(log(1/(1+exp(-thetaX)))-y)**2
        interValue2=(log(1/(1+exp(-thetaX)))-y)*x
        yield None, (interValue1,interValue2.tolist(),y)
             
         
     def SumCostDerivative(self,_,values): 
         #J=0
         #dJ=np.zeros(len(values[0][1]))   
         #for i in values:
         #    J=J+i[0]
         #    dJ=dJ+i[1]
         #theta=self.get_theta()
         #thetaArray=np.array([float(i) for i in theta])
         #thetaArray=thetaArray-0.1*dJ
         yield None, values        
          
     def steps(self):
        return [self.mr(mapper=self.CostDerivative,
                        reducer=self.SumCostDerivative)]

if __name__ == '__main__':
    MRCostAndDerivative.run()
    
