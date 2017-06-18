import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from lmfit import minimize, Parameters
def find_modified_plaw_resid_quad(params, x, data, sig_data):
    square = params['squareterm'].value
    normalize = params['normal'].value
    index = params['alpha'].value
    slope = params['slope'].value
    intercept = params['intercept'].value
    model = np.float64( ((x*x*square)+ (x * slope) + intercept)*normalize*np.power(x,(-1.0)*index))
    return (data-model)/sig_data
    
