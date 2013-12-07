import sys
import random

import numpy as np
import pickle
import os
from costDerivative import MRCostAndDerivative


#THETA_FILE="/tmp/emr.logregression.theta"
THETA_FILE="~/Documents/Project-3/theta.txt"
    


if __name__ == '__main__':
    args = sys.argv[1:]
    costDerivativeJob=MRCostAndDerivative(args=args)
    costDerivativeJob.run()
    