import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from lmfit import minimize, Parameters
def f_test(chiratio, dof1, dof2):
    dof_power_term = np.power(dof1/dof2, dof1/2)
    f_ratios_term = np.power(chiratio, 1/(2 * (dof1-2)))/np.power((1 + chiratio*dof1/dof2), 1/(2*(dof1+dof2)))
    gamma_term = sp.special.gamma((dof1+dof2)/2)/(sp.special.gamma(dof1/2)*sp.special.gamma(dof2/2))
    total = dof_power_term * f_ratios_term * gamma_term
    return total
    
