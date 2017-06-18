import numpy as np
def chi_square(model, data, sigmas):
    diff = np.power(np.float64(data-model),2)
    #print model/sigmas
    diff = diff/np.power(sigmas,2)
    #print sigmas**2
    chisq = np.sum(diff)
    #print chisq
    return chisq
