from lmfit import minimize, Parameters
import numpy as np
def get_synchrotron(params, x):
    synnorm = params['synnormal'].value
    synalpha = params['synalpha'].value
    synchrotronmodel = np.array(synnorm*np.power(x, synalpha*(-1.0)))
    return synchrotronmodel
