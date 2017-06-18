#!/usr/bin/python
''' 
program to read in spectra and Stokes parameter spectra for a blazar
model the spectra as a two component model for synchrotron radiation
and a big blue bump.
'''
#import modules#################################################################
import os 
import numpy as np
import scipy as sp
from scipy import misc
import csv
from scipy.signal import convolve, boxcar
from StringIO import StringIO
from astropy.io import fits as pyfits
import matplotlib
import matplotlib.pyplot as plt
import argparse
from astropy.io import fits as pyfits
from scipy.interpolate import interp1d
import lmfit
from lmfit import  Model, minimize, Parameters
from scipy import stats
#import functions I've written##################################################
from correct_extinction import correct_extinction
from readspectrum import readspectrum
from plot_lines_template import plot_lines_template
from find_line import find_line
from find_line_resid import find_line_resid
from rebin import rebin
from plot_q_u_p import plot_q_u_p
from defpol import defpol
from plot_spectra_orig import  plot_spectra_orig
from scipy.optimize import curve_fit as cf
from find_plaw import find_plaw
from find_plaw_resid import find_plaw_resid
from find_plaw_linear_pol import find_plaw_linear_pol
from find_plaw_linear_pol_resid import find_plaw_linear_pol_resid
from twocomponentmodel import twocomponentmodel
from twocomponentmodel_resid import twocomponentmodel_resid
from find_line import find_line
from plot_alpha import plot_alpha
from chi_square import chi_square
from plot_polarized_flux_fit import plot_polarized_flux_fit
from get_order_of_magnitude import get_order_of_magnitude as gom
from bin_array import bin_array as bin_array
from get_synchrotron import get_synchrotron
from get_bigbluebump import get_bigbluebump
from plot_two_component_model import plot_two_component_model
from plot_models import plot_models
from find_modified_plaw import find_modified_plaw
from find_modified_plaw_resid import find_modified_plaw_resid
from plot_polflux_sf import plot_polflux_sf
#End of importing modules#######################################################

# parse the name of the file that has the input info from the command line
parser = argparse.ArgumentParser()
parser.add_argument("indata",  type=str, help="File that Keeps the path to qfiles, ufiles, spectrum, extinction file, redshift, perkins telescope data, and name of source")
args = parser.parse_args()
inputs=np.genfromtxt(args.indata, dtype=None, delimiter=',')
#have the inputs read in, now need to define the output directories.
nameofsource=inputs.item()[6].strip()
dirout = '../outputfiles/'+nameofsource+'/VARBBB/textout/'
plotout = '../outputfiles/'+nameofsource+'/VARBBB/plots/'
bumpoutfile = dirout+'BIG_BLUE_BUMP_NORMAL.txt'
synoutfile = dirout+'SYNCHROTRON_NORMAL.txt'
realbumpoutfile = dirout+'real_big_blue_bump.txt'

#check to see if output directories exist, if not make them.
diroutcheck = os.path.dirname(dirout)
plotoutcheck = os.path.dirname(dirout)
if not os.path.exists(diroutcheck):
    os.makedirs(diroutcheck)
if not os.path.exists(plotoutcheck):
    os.makedirs(plotoutcheck)

oldplotout = plotout
#get the name of the emission line template and plot it
tempdir = os.path.dirname(inputs.item()[0].strip())
tempdir = '../'+nameofsource+'/support_files/'
tempfile = tempdir+'template.txt'
#read the template x, y values of the template in, correct it for galactic reddening
redshift = np.float32(inputs.item()[4])
linetemplate= np.genfromtxt(tempfile, dtype=None)
y = correct_extinction(inputs.item()[3].strip(), linetemplate[:,0]*(1.+redshift), linetemplate[:,1])
x = linetemplate[:, 0]
c = np.float64(2.99792458e+18) #angstrom/s
xnu = np.float64(c/x)
plot_lines_template(x, y, xnu, plotout)
xobs = x*(1.0+redshift)

'''
Begin the main loop of the program.  Will read in the q, u, and flux spectra.
subtract the template.  Determine the polarized flux  and fit the models' 
'''
qspectra = np.genfromtxt(inputs.item()[0].strip(), dtype='S100')
uspectra = np.genfromtxt(inputs.item()[1].strip(), dtype='S100')
fspectra = np.genfromtxt(inputs.item()[2].strip(), dtype ='S100')
size = len(qspectra)

''' Create some arrays that will be populated inside the main loop of the 
program.'''
################################################################################
i = 0
alpha_array = np.array([])
mjdarr = np.array([])
alpha_err = np.array([])
bbbnormalize = np.array([])
synchrotron_normalize = np.array([])
bbb_integratedflux= np.array([])
synchrotron_integratedflux=np.array([])
total_integratedflux = np.array([])
polarized_flux_sf = np.array([])
syn_bbb_models_sf = np.array([])
################################################################################

for i in range(size):
# Read the spectrum into an array
    (qspec, qerr, qend, qheader, qmjd) = readspectrum(qspectra[i])
    (uspec, uerr, uend, uheader, umjd) = readspectrum(uspectra[i])
    (fspec, ferr, fend, fheader, fmjd) = readspectrum(fspectra[i])
    mjdarr = np.append(mjdarr, qmjd)
    plotout = oldplotout+'/'+str(qmjd)+'/'
    if not os.path.exists(plotout):
        os.makedirs(plotout)
    #create the array for the wavelength
    wavend = fend*4.0 + 4000.0
    wave = np.linspace(4000, wavend,fend)
    origcorrected = correct_extinction(inputs.item()[3].strip(), wave, fspec)
    #subtract the emission lines template
    restwave = wave/(1.0+redshift)
    restnu = c/restwave
    interpfunc=interp1d(x, y, kind='cubic', bounds_error=False)
    newtemplate= interpfunc(restwave)
    thisspec = origcorrected-newtemplate
    specerr = (ferr/c)*restwave**2
    nuspec = np.float64((thisspec/c)*np.power(restwave,2))
    nuorig = np.float64((origcorrected/c)*np.power(restwave,2))
    plot_spectra_orig(origcorrected, thisspec, nuorig, nuspec, restwave, restnu, plotout, qmjd)
    if qspec.shape != fspec.shape:
       qspec = rebin(qspec, fspec.shape)
       print 'rebinned q'
    if uspec.shape[0] != fspec.shape[0]:
       uspec = rebin(uspec, fspec.shape)
       print 'rebinned u' 
    #correct polarization for statistical bias (Wardle and Kronberg, 1974)
    #p = np.sqrt(qspec**2 + uspec**2)
    # bin q and u in order to determine polarization and the direction of polarization
    boxcarwindow = 15
    p, sp, thetp, sthetp, qspec, sqspec, uspec, suspec, oldq, oldu = defpol(qspec, uspec, boxcarwindow)    
    #bin the q, u, and flux arrays
    numberofbins = len(nuorig)/boxcarwindow
    polarflux, spflux = bin_array(nuorig, numberofbins)
    polarnu, spolarnu = bin_array(restnu, numberofbins)    
    fitflux, sfitflux = bin_array(nuspec, numberofbins)
    polarflux = polarflux*p
   #spflux = np.sqrt((polarflux*sp)**2 + (spflux*p)**2) * np.sqrt(boxcarwindow)
    polarwave = c/polarnu
    plot_q_u_p(qspec, sqspec, uspec, suspec, p, sp, thetp, sthetp, polarnu, polarwave, plotout, qmjd, oldq, oldu, restnu)
    normflux = 10.0**np.fix(np.log10(np.mean(polarflux)))
    normnu = 10.0**np.fix(np.log10(np.mean(polarnu)))

#################### Fit the polarized flux with a power-law####################
##### use the lmfit package to determine alpha##################################
    plaw_params = Parameters() 
    initialnormal = np.float64(np.median(polarflux)*np.median(polarnu))/normflux
    plaw_params.add('norm', value=initialnormal, min=np.float64(0.0))
    plaw_params.add('alpha', value=np.float64(1.5))
    output = minimize(find_plaw_resid, plaw_params, args=(polarnu, polarflux/normflux, 4.0*spflux/normflux))
    model = find_plaw(plaw_params, polarnu)*normflux
    lmfit.printfuncs.report_fit(output.params)
    name='justplaw'
    plot_polarized_flux_fit(polarflux, spflux, polarnu, qmjd, plotout, restnu, oldq, oldu, nuorig, model, name)
    ci = lmfit.conf_interval(output, maxiter=1000)
#################################################################################    

########### Fit the polarized flux with a modified power-law###################
##########  F = (A * nu + B) * Norm * nu^(-alpha)##############################
########## Start with fitting the polarization with P = (A nu + B)##############
    linepol_params = Parameters()
    linepol_params.add('slope', value = 1/gom(np.median(restnu)))
    linepol_params.add('intercept', value = 1.0)
    linepol_output = minimize(find_line_resid, linepol_params, args=(polarnu, p, sp))
    linepol_model = find_line(linepol_params, polarnu)
    ### Use the Slope and Intercept as fixed paramters for the modified power-law
    modified_plaw_params = Parameters()
    minslope = linepol_params['slope'].value - 2*linepol_params['slope'].stderr
    maxslope = linepol_params['slope'].value + 2*linepol_params['slope'].stderr
    minint = linepol_params['intercept'].value - 2*linepol_params['intercept'].stderr
    maxint = linepol_params['intercept'].value + 2*linepol_params['intercept'].stderr
    modified_plaw_params.add('slope', value = linepol_params['slope'].value, min=minslope, max=maxslope)
    modified_plaw_params.add('intercept', value= linepol_params['intercept'].value, min=minint, max=maxint)
    modified_plaw_params.add('normal', value = plaw_params['norm'].value, min=np.float64(0.0))
    modified_plaw_params.add('alpha', value =  plaw_params['alpha'].value)
    modified_plaw_output = minimize(find_modified_plaw_resid, modified_plaw_params, args=(polarnu, polarflux/normflux, 4.0*spflux/polarflux))
    modified_plaw_model = find_modified_plaw(modified_plaw_params, polarnu)*normflux
    name='modified_plaw'
    plot_polarized_flux_fit(polarflux, spflux, polarnu, qmjd, plotout, restnu, oldq, oldu, nuorig, modified_plaw_model, name)
###############################################################################
# Perform an F-test on the two models for the polarized flux###################
    chisquare1 = chi_square(model, polarflux, 4.0*spflux)
    chisquare2 = chi_square(modified_plaw_model, polarflux, 4.0*spflux)
    chiratio = chisquare1/chisquare2
    dof1 = len(model) - 2
    dof2 = len(model) - 4
    sf_polarized_flux = stats.f.sf(chiratio, dof1, dof2)
    polarized_flux_sf = np.append(polarized_flux_sf, sf_polarized_flux)
################################################################################
    normalflux = gom(fitflux)
    #create the parameters for the two component model
    twocomp_params = Parameters()  
    #perform the fit for the two component model.
    #first select the windows for the fit
    fitwindow = np.array([])
    fitwindowa = np.where((polarnu > 6.0e+14) &( polarnu < 7.2e+14)) 
    fitwindowb = np.where((polarnu > 9.4e+14) & (polarnu < 10.2e+14))
    fitwindow = np.append(fitwindow, fitwindowa)
    fitwindow = np.append(fitwindow, fitwindowb)
    fitwindow = fitwindow.astype(int)
    abegin = np.mean(polarnu/normnu)*np.mean(fitflux/normalflux)/3.
    bbegin = np.mean(fitflux)/3.*np.mean(polarnu/normnu)**(-1/3)
    twocomp_params.add('synnormal', value =abegin, min = np.float64(0))
    twocomp_params.add('bumpnormal', value = bbegin, min=np.float64(0))
    twocomp_params.add('alpha', value = plaw_params['alpha'].value , vary=False)
    twocomp_params.add('bumpindex', value = np.float32(1./3.), vary=False)
    #get initial values for parameters with  Nelder-Mead
    model_for_spectrum = lmfit.minimize(twocomponentmodel_resid, twocomp_params, args=(polarnu[fitwindow]/normnu, fitflux[fitwindow]/normalflux, 4.0*sfitflux[fitwindow]/normalflux),method='Nelder')
    #Now solve with Levenberg-Marquadt
    model_for_spectrum = minimize(twocomponentmodel_resid, twocomp_params, args=(polarnu[fitwindow]/normnu, fitflux[fitwindow]/normalflux, 4.0*sfitflux[fitwindow]/normalflux))
################################################################################

    #generate the models, get chi-square, and integrate the total flux
    #ci = lmfit.conf_interval(model_for_spectrum, maxiter=1000)
    mymodel = twocomponentmodel(twocomp_params, restnu/normnu)*normalflux
    mydatamodel = twocomponentmodel(twocomp_params, polarnu/normnu)*normalflux
    synmodel = get_synchrotron(twocomp_params, restnu/normnu)*normalflux
    bumpmodel = get_bigbluebump(twocomp_params, restnu/normnu)*normalflux
    model_chisq = chi_square(mydatamodel[fitwindow], fitflux[fitwindow], 4.0*sfitflux[fitwindow])
    model_reduced_chi = model_chisq/(np.float(len(fitwindow)) - 2)
    df1 = (np.float(len(fitwindow)) - 2)
    modelflux = np.trapz(mymodel, restnu)*(-1)
    modelsyn = np.trapz(synmodel, restnu)*(-1)
    modelbump = np.trapz(bumpmodel, restnu)*(-1)
#    print modelbump, modelsyn, modelflux, 'fits'
    bbb_integratedflux= np.append(bbb_integratedflux, modelbump)
    synchrotron_integratedflux=np.append(synchrotron_integratedflux, modelsyn)
    total_integratedflux = np.append(total_integratedflux, modelflux)
 
 
 ###############################################################################
 ################Perform the two-component fit witht the alpha obtained########
 ################# By fitting the polarized flux with the modified Plaw#######
 
    twocomp_params_2 = Parameters()  
    twocomp_params_2.add('synnormal', value =abegin, min = np.float64(0))
    twocomp_params_2.add('bumpnormal', value = bbegin, min=np.float64(0))
    twocomp_params_2.add('alpha', value = modified_plaw_params['alpha'].value , vary=False)
    twocomp_params_2.add('bumpindex', value = np.float32(1./3.), vary=False)
     #get initial values for parameters with  Nelder-Mead
    model_for_spectrum = lmfit.minimize(twocomponentmodel_resid, twocomp_params_2, args=(polarnu[fitwindow]/normnu, fitflux[fitwindow]/normalflux, 4.0*sfitflux[fitwindow]/normalflux),method='Nelder')
    #Now solve with Levenberg-Marquadt
    model_for_spectrum = minimize(twocomponentmodel_resid, twocomp_params_2, args=(polarnu[fitwindow]/normnu, fitflux[fitwindow]/normalflux, 4.0*sfitflux[fitwindow]/normalflux))
    mymodel_2 = twocomponentmodel(twocomp_params_2, restnu/normnu)*normalflux
    mydatamodel_2 = twocomponentmodel(twocomp_params_2, polarnu/normnu)*normalflux
    synmodel_2 = get_synchrotron(twocomp_params_2, restnu/normnu)*normalflux
    bumpmodel_2 = get_bigbluebump(twocomp_params_2, restnu/normnu)*normalflux

#####################Perform an F-Test on each of the two component fits########
    chisq1 = chi_square(mydatamodel[fitwindow], fitflux[fitwindow], 4.0*sfitflux[fitwindow])
    chisq2 = chi_square(mydatamodel_2[fitwindow], fitflux[fitwindow], 4.0*sfitflux[fitwindow])
    chiratio = chisq1/chisq2
    dof1 = len(fitwindow) - 3
    dof2 = len(fitwindow) - 5
    sf_syn_bbb_models = stats.f.sf(chiratio, dof1, dof2)
    syn_bbb_models_sf = np.append(syn_bbb_models_sf, sf_syn_bbb_models)
 ###############################################################################  
###############################################################################
    #############Perform a fit of a power-law to the spectrum##################
    ########### Do an F-Test with this fit and the 2 component model###########
    just1plaw_params = Parameters() 
    normalizeplawinit = np.mean((fitflux[fitwindow]/normalflux)**(2./3.))
    just1plaw_params.add('norm', value=normalizeplawinit, min=np.float64(0.0))
    just1plaw_params.add('alpha', value=np.float64(-1./3))
    just1plaw_output = minimize(find_plaw_resid, just1plaw_params, args=(polarnu[fitwindow], fitflux[fitwindow]/normalflux, 4.0*sfitflux[fitwindow]/normalflux))
    
    just1plaw_model = find_plaw(just1plaw_params, polarnu)*normalflux
    print just1plaw_params
    just1plaw_chisq = chi_square(just1plaw_model[fitwindow], fitflux[fitwindow], 4.0*sfitflux[fitwindow])
    just1plaw_reduced_chi = just1plaw_chisq/(np.float(len(fitwindow)) - 2)
    df2 = (np.float(len(fitwindow)) - 2)
    chiratio =  just1plaw_reduced_chi/model_reduced_chi
    print 'chisq ratio', chiratio
    print 'justplawchi', just1plaw_chisq
    print 'twocomp chi', model_chisq
    print 'df1 , df2', df1, df2
    ftest = stats.f.sf(chiratio, df1, df2)
    print 'f-test', ftest
    print 'mjd', qmjd
#############################################################################
############Plot the best fit 2 component models along with the best powerlaw
    name = '_bestfitmodels.png'
    plot_two_component_model(restnu, normnu, nuspec, normalflux, polarnu, fitflux, sfitflux, mymodel_2, synmodel_2, bumpmodel_2, just1plaw_model, plotout, qmjd, name)
    name = '_bestfitmodels_2.png'
    plot_two_component_model(restnu, normnu, nuspec, normalflux, polarnu, fitflux, sfitflux, mymodel, synmodel, bumpmodel, just1plaw_model, plotout, qmjd, name)
    bbbnormalize = np.append(bbbnormalize, twocomp_params['bumpnormal'].value)
    synchrotron_normalize = np.append(synchrotron_normalize, twocomp_params['synnormal'].value)
    alpha_array = np.append(alpha_array, twocomp_params['alpha'].value)
  #  print twocomp_params
    #print the confidence intervals for the fit parameters
    #ci = lmfit.conf_interval(model_for_spectrum, maxiter=1000)
    #lmfit.printfuncs.report_ci(ci)
    #print model
    #print params['norm'].value, 'fitnormal', params['alpha'].value, 'alpha'
    #themodel = result.best_fit
    #################################################
    alpha_array = np.append(alpha_array, plaw_params['alpha'].value)
    alpha_err = np.append(alpha_err, plaw_params['alpha'].stderr)


plot_polflux_sf(mjdarr, polarized_flux_sf, oldplotout, 'polarized_flux_models_ftest')

#plot how parameters of interest have changed throughout several years
print len(alpha_array), len(alpha_err), len(mjdarr)
plot_alpha(mjdarr, alpha_array, alpha_err, oldplotout)
plot_models(mjdarr, bbb_integratedflux, synchrotron_integratedflux, total_integratedflux, oldplotout)
alphaoutput = dirout+'alphaoutput.csv'
with open(alphaoutput, 'wb') as csvfile:
   spamwriter = csv.writer(csvfile, delimiter=',')
   csvfile.write(' MJD, alpha, sigma \n')
   for x in range(len(mjdarr)):
      spamwriter.writerow([mjdarr[x], alpha_array[x], alpha_err[x]])


