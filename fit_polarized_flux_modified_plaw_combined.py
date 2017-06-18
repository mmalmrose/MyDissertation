import lmfit
from lmfit import  Model, minimize, Parameters
from plot_polarized_flux_fit_combined import  plot_polarized_flux_fit_combined
from find_plaw_resid import find_plaw_resid
from find_plaw import find_plaw
import numpy as np
from find_line_resid import find_line_resid
from find_line import find_line
from find_modified_plaw import find_modified_plaw
from get_order_of_magnitude import get_order_of_magnitude as gom
from find_modified_plaw_resid import find_modified_plaw_resid
def fit_polarized_flux_modified_plaw_combined(restnu, polarnu, polarflux, spflux, fitwindow, normflux, plotout, qmjd, nuorig, p, sp, plaw_params, polspec):
########### Fit the polarized flux with a modified power-law###################
##########  F = (A * nu + B) * Norm * nu^(-alpha)##############################
########## Start with fitting the polarization with P = (A nu + B)##############
    linepol_params = Parameters()
    linepol_params.add('slope', value = 1.0/gom(np.median(restnu)))
    linepol_params.add('intercept', value = 1.0)
    linepol_output = minimize(find_line_resid, linepol_params, 
    args=(polarnu, p, sp))
    linepol_model = find_line(linepol_params, polarnu)
    ### Use the Slope and Intercept as fixed paramters for the modified power-law
    modified_plaw_params = Parameters()
    minslope = linepol_params['slope'].value - 3*linepol_params['slope'].stderr
    maxslope = linepol_params['slope'].value + 3*linepol_params['slope'].stderr
    minint = linepol_params['intercept'].value - 3*linepol_params['intercept'].stderr
    maxint = linepol_params['intercept'].value + 3*linepol_params['intercept'].stderr
    modified_plaw_params.add('slope', value = linepol_params['slope'].value, 
    min=minslope, max=maxslope)
    modified_plaw_params.add('intercept', value= 
    linepol_params['intercept'].value, min=minint, max=maxint)
    modified_plaw_params.add('normal', value = plaw_params['norm'].value, 
    min=np.float64(0.0))
    modified_plaw_params.add('alpha', value =  plaw_params['alpha'].value)
    modified_plaw_output = minimize(find_modified_plaw_resid, 
    modified_plaw_params, args=(polarnu[fitwindow], polarflux[fitwindow]/normflux, 
    spflux[fitwindow]/polarflux[fitwindow]), method = 'nelder')
    modified_plaw_output = minimize(find_modified_plaw_resid, 
    modified_plaw_params, args=(polarnu[fitwindow], polarflux[fitwindow]/normflux, 
    spflux[fitwindow]/polarflux[fitwindow]))    
    modified_plaw_model = find_modified_plaw(modified_plaw_params, polarnu)*normflux
    name='modified_plaw'
    plot_polarized_flux_fit_combined(polarflux, spflux, polarnu, qmjd, plotout, restnu, 
    nuorig, modified_plaw_model, name, polspec)
    return linepol_params, linepol_output, linepol_model, modified_plaw_params, modified_plaw_model
