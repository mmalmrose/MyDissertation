from lmfit import minimize, Parameters
import numpy as np
def get_bigbluebump(params, x):
    bumpnorm = params['bumpnormal'].value
    bumpalpha = params['bumpindex'].value
    bigbluebumpmodel = np.array(bumpnorm*np.power(x, bumpalpha*(-1.0)))
    return bigbluebumpmodel
