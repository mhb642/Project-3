# generate first theta
import numpy as np
import pickle
from numpy import random

THETA_FILE="/tmp/emr.theta"
theta=random.uniform(0,1,2)
#f=open(THETA_FILE,'w')
#pickle.dump(theta,f)
#f.close()
np.savetxt(THETA_FILE,theta,delimiter=',')