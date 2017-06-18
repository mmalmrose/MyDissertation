#################### Fit the polarized flux with a power-law####################
##### use the lmfit package to determine alpha##################################
#############DEFINE THE FREQUENCIES FOR WHICH THE FIT WILL BE PERFORMED#########
###############################################################################
import lmfit
from lmfit import  Model, minimize, Parameters
from plot_polarized_flux_fit import  plot_polarized_flux_fit
from find_plaw import find_plaw
from find_plaw_resid import find_plaw_resid
import numpy as np
from matplotlib import pyplot as plt
def fit_polarized_flux_plaw(polarnu, polarflux, normflux, spflux, initialnormal,
    fitwindow, qmjd, plotout, restnu, oldq, oldu, nuorig, normnu):
    plaw_params = Parameters() 
    #fitwindow = np.where((polarnu > 5.1e+14) &( polarnu < 8e+14))  
    #fitwindow = np.where((polarnu > 4.1e+14) &( polarnu < 7.5e+14)) #3C273
    #fitwindow = np.where((polarnu > 8.0e+14) &( polarnu < 1.2e+15))   #OJ248 
    #fitwindow = np.where((polarnu > 6.0e+14) &( polarnu < 1.e+15)) #3C279
    #fitwindow = np.where((polarnu > 8.0e+14) &( polarnu < 1.25e+15)) #CTA102
    #fitwindow = np.where((polarnu > 7.0e+14) &( polarnu < 1.35e+15))  #3C454.3
    #fitwindow = np.where((polarnu > 5.0e+14) &( polarnu < 9.05e+14))  #0735+178
    #fitwindow = np.where((polarnu > 4.0e+14) &( polarnu < 7.0e+14))  #BL LAC   
  #  print '======================INITIAL NORMAL================================='
  #  print initialnormal
  #  print normflux
  #  print normnu
  #  print'======================================================================'
    goodfit = np.where(polarflux[fitwindow] > 0.0)
    print goodfit, 'goodfit'
    #plt.plot(polarnu[fitwindow], polarflux[fitwindow], 'o')
    #plt.plot(polarnu, polarflux)
    #plt.plot(polarnu[fitwindow], polarflux[fitwindow], 's')
    #plt.show()
    plaw_params.add('norm', value=initialnormal, min=np.float64(0.0))
    plaw_params.add('alpha', value=np.float64(0.0))
    output = minimize(find_plaw_resid, plaw_params, args=(polarnu[fitwindow]/normnu, 
    polarflux[fitwindow]/normflux, spflux[fitwindow]/normflux), method='nelder')
  #  print 'plaw params', plaw_params
    output = minimize(find_plaw_resid, plaw_params, args=(polarnu[fitwindow]/normnu, 
    polarflux[fitwindow]/normflux, spflux[fitwindow]/normflux))
   # print 'plaw params', plaw_params    
    model = find_plaw(plaw_params, polarnu/normnu)*normflux
   # print model
   # print polarflux
    lmfit.printfuncs.report_fit(output.params)
    name='plaw'
    plot_polarized_flux_fit(polarflux, spflux, polarnu, qmjd, plotout, restnu, 
    oldq, oldu, nuorig, model, name)
   # print 'normflux', normflux
    return plaw_params, output, model
