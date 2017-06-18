import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from lmfit import minimize, Parameters
def find_plaw_linear_pol(params, x):
    normalize = params['norm'].value
    index = params['alpha'].value
    intercept = params['b'].value
    slope = params['m'].value
    model = np.float64(normalize*(slope*x + intercept)*(x**((-1)*index)))
    return model
    
