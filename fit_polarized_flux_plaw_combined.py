#################### Fit the polarized flux with a power-law####################
##### use the lmfit package to determine alpha##################################
#############DEFINE THE FREQUENCIES FOR WHICH THE FIT WILL BE PERFORMED#########
###############################################################################
import lmfit
from lmfit import  Model, minimize, Parameters
from plot_polarized_flux_fit_combined import  plot_polarized_flux_fit_combined
from find_plaw import find_plaw
from find_plaw_resid import find_plaw_resid
import numpy as np
def fit_polarized_flux_plaw_combined(polarnu, polarflux, normflux, spflux, initialnormal,
    fitwindow, qmjd, plotout, restnu,  nuorig, polspec):
    plaw_params = Parameters() 
    plaw_params.add('norm', value=initialnormal, min=np.float64(0.0))
    plaw_params.add('alpha', value=np.float64(1.5))
    output = minimize(find_plaw_resid, plaw_params, args=(polarnu[fitwindow], 
    polarflux[fitwindow]/normflux, spflux[fitwindow]/normflux), method='nelder')
    print 'plaw params', plaw_params
    output = minimize(find_plaw_resid, plaw_params, args=(polarnu[fitwindow], 
    polarflux[fitwindow]/normflux, spflux[fitwindow]/normflux))
    print 'plaw params', plaw_params    
    model = find_plaw(plaw_params, polarnu)*normflux
    lmfit.printfuncs.report_fit(output.params)
    name='plaw'
    plot_polarized_flux_fit_combined(polarflux, spflux, polarnu, qmjd, plotout, restnu, 
     nuorig, model, name, polspec)
    return plaw_params, output, model
