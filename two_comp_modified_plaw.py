import lmfit
from lmfit import  Model, minimize, Parameters
from scipy import stats
import numpy as np
from twocomponentmodel_resid import twocomponentmodel_resid
from twocomponentmodel import twocomponentmodel
from get_synchrotron import get_synchrotron
from get_bigbluebump import get_bigbluebump
def two_comp_modified_plaw(fitflux, sfitflux, normalflux, polarnu, normnu, plaw_params, twocomp_params, restnu, modified_plaw_params, fitwindow, qmjd, normflux):
    twocomp_params_2 = Parameters()  
    abegin = np.float64((np.mean(fitflux/normalflux)/2)*np.mean(polarnu/normnu)**((-1.0)*plaw_params['alpha'].value))
    print abegin
    bbegin = (np.mean(fitflux/normalflux)/2.)*np.mean(polarnu/normnu)**(-1/3)     
    minbumpflux = np.float64(1.84644587168e-12)
    integrated = np.max(restnu)**((-1)*twocomp_params['bumpindex'].value
    +1.0) - np.min(restnu)**((-1)*twocomp_params['bumpindex'].value+1.0)  
    minbumpnorm =  minbumpflux *((-1)*twocomp_params['bumpindex'].value+1.0
    )*np.power(normnu, (-1)*twocomp_params['bumpindex'].value)/integrated
    minbumpnorm = minbumpnorm/normalflux
    twocomp_params_2.add('synnormal', value = abegin, min = np.float64(0))
    #twocomp_params_2.add('bumpnormal', value = bbegin, min = minbumpnorm)
    twocomp_params_2.add('synalpha', value = modified_plaw_params['alpha'].value, 
    vary=False)
    twocomp_params_2.add('bumpindex', value = np.float64(-1./3.), vary=False)
    if np.absolute(qmjd - 56714.0) < 1.0:
        aaa= 3.1430305928799E-17/(np.power(normnu, (0.77))*normflux)
        twocomp_params_2.add('bumpnormal', value=aaa, vary=False)
    else:
        twocomp_params_2.add('bumpnormal', value = bbegin, min=minbumpnorm)
    #Now solve with Nelder-Mead    
     #get initial values for parameters with  Nelder-Mead
    model_for_spectrum = lmfit.minimize(twocomponentmodel_resid, 
    twocomp_params_2, args=(polarnu[fitwindow]/normnu, 
    fitflux[fitwindow]/normalflux, sfitflux[fitwindow]/normalflux),
    method='Nelder')
    #Now solve with Levenberg-Marquadt
    model_for_spectrum = minimize(twocomponentmodel_resid, twocomp_params_2, 
    args=(polarnu[fitwindow]/normnu, fitflux[fitwindow]/normalflux, 
    sfitflux[fitwindow]/normalflux))
    mymodel_2 = twocomponentmodel(twocomp_params_2, restnu/normnu)*normalflux
    mydatamodel_2 = twocomponentmodel(twocomp_params_2, polarnu/normnu)*normalflux
    synmodel_2 = get_synchrotron(twocomp_params_2, restnu/normnu)*normalflux
    bumpmodel_2 = get_bigbluebump(twocomp_params_2, restnu/normnu)*normalflux
    modelflux_2 = np.trapz(mymodel_2, restnu)*(-1)
    modelsyn_2 = np.trapz(synmodel_2, restnu)*(-1)
    modelbump_2 = np.trapz(bumpmodel_2, restnu)*(-1)
    return twocomp_params_2, model_for_spectrum, mymodel_2, mydatamodel_2, synmodel_2, bumpmodel_2, modelflux_2, modelsyn_2, modelbump_2
