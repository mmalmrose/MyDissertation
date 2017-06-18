import numpy as np
from blackbody import blackbody as bb
from lmfit import minimize, Parameters
def twocomponentmodel_bb_resid(params, x, xnorm, data ,sigmadata):
    synnormal = params['synnormal'].value
    alpha = params['synalpha'].value
    bumpnormal = params['bumpnormal'].value
    bumpindex = params['bumpindex'].value
    bbnormal = params['blackbody_norm'].value
    bbtemp = params['temperature'].value
    bbody = bb(x*xnorm, bbtemp)
    synchrotron = np.float64(synnormal)*np.power(x,alpha*(-1.0))
    bigbluebump = np.float64(bumpnormal)*np.power(x,bumpindex*(-1.0))
    model = synchrotron+bigbluebump
    resid = (data-model)/sigmadata
    return resid
