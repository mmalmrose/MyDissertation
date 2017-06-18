import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from lmfit import minimize, Parameters
def find_plaw_linear_pol_resid(params, x, data, sigmadata):
    normalize = params['norm'].value
    index = params['alpha'].value
    intercept = params['b'].value
    model = np.float64((normalize*x + intercept)*(x**((-1)*index)))
    resid = (data-model)/sigmadata
    return resid
    
