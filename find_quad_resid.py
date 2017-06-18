import numpy as np
import scipy as sp
def find_quad_resid(params, x, data, sig_data):
    square = params['squareterm'].value
    slope = params['slope'].value
    intercept = params['intercept'].value
    model = np.float64((square*x*x) + (slope*x) + intercept)
    #if data is none:
    #    return model
    #if sig_data is none:
    #    return (data-model)
    return (data - model)/sig_data
    
