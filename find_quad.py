import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from lmfit import minimize, Parameters
def find_quad(params, x):
    square = params['squareterm'].value
    slope = params['slope'].value
    intercept = params['intercept'].value
    model = np.float64((square*x*x) + (slope*x) + intercept)
    return model
