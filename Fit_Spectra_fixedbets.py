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
from find_line_resid import find_line_resid
from plot_chisquare import plot_chisquare
from plot_flux_onlyplaw import plot_flux_onlyplaw 
from twocomponentmodel_BB_resid import twocomponentmodel_BB_resid
from blackbody import blackbody as bb
from fit_polarized_flux_plaw import fit_polarized_flux_plaw
from fit_polarized_flux_modified_plaw import fit_polarized_flux_modified_plaw
from fit_flux_two_comp_model import fit_flux_two_comp_model
from fit_flux_two_comp_wbb import fit_flux_two_comp_wbb
from two_comp_modified_plaw import two_comp_modified_plaw
from two_comp_modified_plaw_fixedbbb import two_comp_modified_plaw_fixedbbb
from just_single_plaw import just_single_plaw
from plot_the_final_models import plot_the_final_models
from fit_polarized_flux_modified_plaw_quad import fit_polarized_flux_modified_plaw_quad
from two_comp_modified_plaw_quad import two_comp_modified_plaw_quad
#End of importing modules#######################################################

# parse the name of the file that has the input info from the command line
parser = argparse.ArgumentParser()
parser.add_argument("indata",  type=str, 
help="File that Keeps the path to qfiles, ufiles, spectrum, extinction file, redshift, perkins telescope data, and name of source")
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
#read the template x, y values of the template in, 
#correct it for galactic reddening
redshift = np.float32(inputs.item()[4])
linetemplate= np.genfromtxt(tempfile, dtype=None)
y = correct_extinction(inputs.item()[3].strip(), linetemplate[:,0]*(1.+redshift)
, linetemplate[:,1])
y1 = linetemplate[:,1]
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
onlyplaw_alpha = np.array([])
sigma_onlyplaw_alpha = np.array([])
modified_alpha_array = np.array([])
sigma_modified_alpha_array = np.array([])
mjdarr = np.array([])
alpha_err = np.array([])
bbbnormalize = np.array([])
synchrotron_normalize = np.array([])
bbb_integratedflux= np.array([])
synchrotron_integratedflux=np.array([])
total_integratedflux = np.array([])
bbb_integratedflux2= np.array([])
synchrotron_integratedflux2=np.array([])
total_integratedflux2 = np.array([])
polarized_flux_sf = np.array([])
#syn_bbb_models_sf = np.array([])
ftest_2comp_model= np.array([])
ftest_2comp_model_mplaw= np.array([])
chi1arr=np.array([])
chi2arr=np.array([])
chi3arr=np.array([])    
bbb_integratedflux_fixedbbb= np.array([])
synchrotron_integratedflux_fixedbbb= np.array([])
total_integratedflux_fixedbbb = np.array([])
chi4arr=np.array([])
mybbbindex=np.array([])
blackbodyflux = np.array([])
temperature_array = np.array([])
bbsynmodel = np.array([])
bbbumpmodel = np.array([])  
totalfluxwbb = np.array([])
blackbodychisquare = np.array([])
polarized_flux_sf2 = np.array([])
synchrotron_integratedflux_quad=np.array([])
total_integratedflux_quad = np.array([])
bbb_integratedflux_quad= np.array([])
chi5arr = np.array([])
linearpol_alpha = np.array([])
linearpol_alpha_sigma = np.array([])
quadratic_alpha = np.array([])
quadratic_alpha_sigma = np.array([])
totalfluxwbb_qp = np.array([])
bbsynmodel_qp =np.array([])
bbbumpmodel_qp = np.array([])
blackbodyflux_qp =np.array([])
temperature_array_qp = np.array([])
blackbodychisquare_qp = np.array([])
totalfluxwbb_quadpol = np.array([])
bbsynmodel_quadpol =  np.array([])
bbbumpmodel_quadpol =  np.array([])
blackbodyflux_quadpol =  np.array([])
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
 #   origcorrected   = fspec
    #subtract the emission lines template
    restwave = wave/(1.0+redshift)
    restnu = c/restwave
    interpfunc=interp1d(x, y, kind='cubic', bounds_error=False)
    newtemplate= interpfunc(restwave)
    thisspec = origcorrected-newtemplate
    specerr = (ferr/c)*restwave**2
    nuspec = np.float64((thisspec/c)*np.power(restwave,2))
    nuorig = np.float64((origcorrected/c)*np.power(restwave,2))
    plot_spectra_orig(origcorrected, thisspec, nuorig, nuspec, restwave, restnu
    , plotout, qmjd)
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
    p, sp, thetp, sthetp, qspec, sqspec, uspec, suspec, oldq, oldu = defpol(
    qspec, uspec, boxcarwindow)    
    #bin the q, u, and flux arrays
    numberofbins = len(nuorig)/boxcarwindow
    polarflux, spflux = bin_array(nuorig, numberofbins)
    polarnu, spolarnu = bin_array(restnu, numberofbins)    
    fitflux, sfitflux = bin_array(nuspec, numberofbins)
    sfitflux = sfitflux * np.sqrt(boxcarwindow)
    polarflux = polarflux*p
    spflux = np.sqrt((polarflux*sp)**2 + (spflux*p)**2) * np.sqrt(boxcarwindow)
    polarwave = c/polarnu
    plot_q_u_p(qspec, sqspec, uspec, suspec, p, sp, thetp, sthetp, polarnu, 
    polarwave, plotout, qmjd, oldq, oldu, restnu)
    normflux = 10.0**np.fix(np.log10(np.mean(polarflux)))
    normnu = 10.0**np.fix(np.log10(np.mean(polarnu)))
#################### Fit the polarized flux with a power-law####################
##### use the lmfit package to determine alpha##################################
#############DEFINE THE FREQUENCIES FOR WHICH THE FIT WILL BE PERFORMED#########
###############################################################################
    fitwindow = np.array([])
    fitwindowa = np.where((polarnu > 6.2e+14) &( polarnu < 6.85e+14)) 
    fitwindowb = np.where((polarnu > 9.08e+14) & (polarnu < 10.2e+14))
    fitwindow = np.append(fitwindow, fitwindowa)
    fitwindow = np.append(fitwindow, fitwindowb)
    fitwindow = fitwindow.astype(int)
################################################################################
#######################Fit the polarized flux with a power law##################
    initialnormal = np.float64(np.median(polarflux)*np.median(polarnu))/normflux
    plaw_params, output, model = fit_polarized_flux_plaw(polarnu, polarflux, 
    normflux, spflux, initialnormal, fitwindow, qmjd, plotout, restnu, oldq,
     oldu, nuorig)
      
########### Fit the polarized flux with a modified power-law###################
##########  F = (A * nu + B) * Norm * nu^(-alpha)##############################
########## Start with fitting the polarization with P = (A nu + B)##############
    linepol_params, linepol_output, linepol_model, modified_plaw_params,modified_plaw_model =fit_polarized_flux_modified_plaw(restnu, polarnu, 
    polarflux, spflux, fitwindow, normflux, plotout, qmjd, oldq, oldu, 
    nuorig, p, sp, plaw_params)
    
########### Fit the polarized flux with a modified power-law####################
########### of the form F = (A * nu^2 + B*nu + C) * NORM * nu^(-alpha)##########
    quadpol_params, quadpol_output, quadpol_model, modified_plaw_params_quad, modified_plaw_model_quad = fit_polarized_flux_modified_plaw_quad(restnu, polarnu, 
    polarflux, spflux, fitwindow, normflux, plotout, qmjd, oldq, oldu, 
    nuorig, p, sp, plaw_params)

################################################################################
################################################################################
###############################################################################
# Perform an F-test on the two models for the polarized flux###################
    chisquareplaw = chi_square(model, polarflux, spflux)
    chisquarelinear = chi_square(modified_plaw_model, polarflux, spflux)
    chisquarequad = chi_square(modified_plaw_model_quad, polarflux, spflux)
    chiratio1 = chisquareplaw/chisquarelinear
    chiratio2 = chisquareplaw/chisquarequad
    dof1 = len(model) - 2
    dof2 = len(model) - 4
    dof3 = len(model) - 5
    sf_polarized_flux = stats.f.sf(chiratio1, dof1, dof2)
    sf_polarized_flux2 = stats.f.sf(chiratio2, dof1, dof3)
    polarized_flux_sf = np.append(polarized_flux_sf, sf_polarized_flux)
    polarized_flux_sf2 = np.append(polarized_flux_sf2, sf_polarized_flux2)
################################################################################
    normalflux = gom(fitflux)
    twocomp_params = Parameters()  
####################Perform the fit for the two component model#################
    modelflux, modelsyn, synmodel, mymodel, bumpmodel, model_chisq,  model_reduced_chi, twocomp_params, model_for_spectrum, mydatamodel, df1, modelbump =fit_flux_two_comp_model(restnu, polarnu, polarflux, spflux, fitwindow, normflux, plotout, qmjd, oldq, oldu, nuorig, fitflux, plaw_params, normalflux, normnu, sfitflux)
    mybbbindex = np.append(mybbbindex, twocomp_params['bumpindex'].value) 
    bbb_integratedflux= np.append(bbb_integratedflux, modelbump)
    synchrotron_integratedflux=np.append(synchrotron_integratedflux, modelsyn)
    total_integratedflux = np.append(total_integratedflux, modelflux)
    #create the parameters for the two component model
 ###############################################################################
 ###################Perform the Two-Component fit with a BB on top##############
    #perform the fit for the two component model.
    #first select the windows for the fit
    
    blackbodyfluxmodel,  twocomp_BB_params, synmodelwbb, bumpmodelwbb, blackbodymodel, synfluxmodel, bbbumpfluxmodel, mychisquarewbb= fit_flux_two_comp_wbb(restnu, polarnu, polarflux, spflux, fitwindow, normflux, plotout, qmjd, oldq, oldu, nuorig, fitflux, plaw_params, normalflux, normnu, sfitflux, twocomp_params, nuspec)
    totalfluxwbb = np.append(totalfluxwbb, synfluxmodel+bbbumpfluxmodel+blackbodyfluxmodel)
    bbsynmodel = np.append(bbsynmodel, synfluxmodel)
    bbbumpmodel = np.append(bbbumpmodel, bbbumpfluxmodel)  
    blackbodyflux = np.append(blackbodyflux, blackbodyfluxmodel)
    temperature_array = np.append(temperature_array, twocomp_BB_params['temperature'].value)
    blackbodychisquare = np.append(blackbodychisquare, mychisquarewbb)
 ###############################################################################
 ###############################################################################
 ################Perform the two-component fit with the alpha obtained########
 ################# By fitting the polarized flux with the modified Plaw#######
    twocomp_params_2, model_for_spectrum, mymodel_2, mydatamodel_2, synmodel_2,bumpmodel_2, modelflux_2, modelsyn_2, modelbump_2 = two_comp_modified_plaw(fitflux, sfitflux, normalflux, polarnu, normnu, plaw_params, twocomp_params, restnu, modified_plaw_params_quad, fitwindow)   
    bbb_integratedflux2= np.append(bbb_integratedflux2, modelbump_2)
    synchrotron_integratedflux2=np.append(synchrotron_integratedflux2, 
    modelsyn_2)
    total_integratedflux2 = np.append(total_integratedflux2, modelflux_2)


################################################################################
################################################################################
###############################################################################
 ################Perform the two-component fit with the alpha obtained########
 ################# By fitting the polarized flux with the modified Plaw#######
 ################# and holding the big blue bump fixed##########################
    #find the normalization to fix the bbb to
    synmodel_fixedbbb, bumpmodel_fixedbbb, modelsyn_fixedbbb, modelsyn_fixedbbb, modelbump_fixedbbb, modelflux_fixedbbb, mydatamodel_fixedbbb= two_comp_modified_plaw_fixedbbb(fitflux, sfitflux, normalflux, polarnu, normnu, plaw_params, twocomp_params, restnu, modified_plaw_params, fitwindow)
    bbb_integratedflux_fixedbbb= np.append(bbb_integratedflux_fixedbbb, bumpmodel_fixedbbb)
    synchrotron_integratedflux_fixedbbb=np.append(
    synchrotron_integratedflux_fixedbbb, modelsyn_fixedbbb)
    total_integratedflux_fixedbbb = np.append(total_integratedflux_fixedbbb, 
    modelflux_fixedbbb)
    
 ###############################################################################
 ###############################################################################
 ################Perform the two-component fit with the alpha obtained########
 ################# By fitting the polarized flux with the modified Plaw#######    
 ################# with a quadratic term ######################################   
    twocomp_params_quadpol, quad_model_for_spectrum, mymodel_quad, mydatamodel_quad, synmodel_quad, bumpmodel_quad, modelflux_quad, modelsyn_quad, modelbump_quad = two_comp_modified_plaw_quad(fitflux, sfitflux, normalflux, polarnu, normnu, plaw_params, twocomp_params, restnu, modified_plaw_params, fitwindow)
    bbb_integratedflux_quad= np.append(bbb_integratedflux_quad, modelbump_quad)
    synchrotron_integratedflux_quad=np.append(synchrotron_integratedflux_quad, modelsyn_quad)
    total_integratedflux_quad = np.append(total_integratedflux_quad, modelflux_quad)
    print modelbump_quad, 'bumpmodel_quad'
    print modelsyn_quad, 'modelsyn_quad'
    print modelflux_quad, 'modelflux_quad'
    

################################################################################
################################################################################
#####################Try Quadratic Polarization with a blackbody################
##################### Component on top #########################################
    twocomp_params_quadpol.add('alpha', value = twocomp_params_quadpol['synalpha'].value, vary=False)
    blackbodyfluxmodel_quadpol,  twocomp_BB_params_quadpol, synmodelwbb_quadpol, bumpmodelwbb_quadpol, blackbodymodel_quadpol, synfluxmodel_quadpol, bbbumpfluxmodel_quadpol, mychisquarewbb_quadpol= fit_flux_two_comp_wbb(restnu, polarnu, polarflux, spflux, fitwindow, normflux, plotout, qmjd, oldq, oldu, nuorig, fitflux, twocomp_params_quadpol, normalflux, normnu, sfitflux, twocomp_params, nuspec)
    totalfluxwbb_quadpol = np.append(totalfluxwbb_quadpol, synfluxmodel_quadpol + bbbumpfluxmodel_quadpol + blackbodyfluxmodel_quadpol)
    bbsynmodel_quadpol = np.append(bbsynmodel_quadpol, synfluxmodel_quadpol)
    bbbumpmodel_quadpol = np.append(bbbumpmodel_quadpol, bbbumpfluxmodel_quadpol)
    blackbodyflux_quadpol = np.append(blackbodyflux_quadpol, blackbodyfluxmodel_quadpol)
#        blackbodyfluxmodel,  twocomp_BB_params, synmodelwbb, bumpmodelwbb, blackbodymodel, synfluxmodel, bbbumpfluxmodel, mychisquarewbb= fit_flux_two_comp_wbb(restnu, polarnu, polarflux, spflux, fitwindow, normflux, plotout, qmjd, oldq, oldu, nuorig, fitflux, plaw_params, normalflux, normnu, sfitflux, twocomp_params, nuspec)
 #   totalfluxwbb = np.append(totalfluxwbb, synfluxmodel+bbbumpfluxmodel+blackbodyfluxmodel)
  #  bbsynmodel = np.append(bbsynmodel, synfluxmodel)
   # bbbumpmodel = np.append(bbbumpmodel, bbbumpfluxmodel)  
   # blackbodyflux = np.append(blackbodyflux, blackbodyfluxmodel)
  #  temperature_array = np.append(temperature_array, twocomp_BB_params['temperature'].value)
 #   blackbodychisquare = np.append(blackbodychisquare, mychisquarewbb)
    
    
    
################################################################################
################################################################################
###############Perform an F-Test on each of the two component fits##############
    chisq1 = chi_square(mydatamodel[fitwindow], fitflux[fitwindow], sfitflux[fitwindow])
    chisq2 = chi_square(mydatamodel_2[fitwindow], fitflux[fitwindow], sfitflux[fitwindow])
    chiratio = chisq1/chisq2
    dof1 = len(fitwindow) - 3
    dof2 = len(fitwindow) - 5
    redchisq1 = chisq1/dof1
    redchisq2 = chisq2/dof2
    #sf_syn_bbb_models = stats.f.sf(chiratio, dof1, dof2)
    #syn_bbb_models_sf = np.append(syn_bbb_models_sf, sf_syn_bbb_models)
 ###############################################################################  
###############################################################################
    #############Perform a fit of a power-law to the spectrum##################
    ########### Do an F-Test with this fit and each of the 2 component models###########
    just1plaw_model, just1plaw_output, just1plaw_params=just_single_plaw(fitflux, sfitflux, normalflux, polarnu, normnu, plaw_params, twocomp_params, restnu, modified_plaw_params, fitwindow)
    just1plaw_chisq = chi_square(just1plaw_model[fitwindow], fitflux[fitwindow],
    sfitflux[fitwindow])
    just1plaw_reduced_chi = just1plaw_chisq/(np.float(len(fitwindow)) - 2)
    df1 = len(fitwindow) - 3
    df2 = (np.float(len(fitwindow)) - 5)
    df3 = len(fitwindow) -2
    chisq1 = chi_square(mydatamodel[fitwindow], fitflux[fitwindow],
    sfitflux[fitwindow])
    redchisq1 = chisq1/df1
    chisq2 = chi_square(mydatamodel_2[fitwindow], fitflux[fitwindow], 
    sfitflux[fitwindow])
    redchisq2 = chisq2/df2
    chisq3 = chi_square(just1plaw_model[fitwindow], fitflux[fitwindow],  
    sfitflux[fitwindow])
    redchisq3 = chisq1/df3
    chisq4 = chi_square(mydatamodel_fixedbbb[fitwindow], fitflux[fitwindow],  
    sfitflux[fitwindow])
    chisq5 = chi_square(mydatamodel_quad[fitwindow], fitflux[fitwindow], sfitflux[fitwindow])
    redchisq5 = chisq5/(df1-2)
    redchisq4 = chisq4/df3
    chiratio1 =  chisq1/chisq3
    chiratio2 = chisq2/chisq3
    chiratio3 = chisq2/chisq4
    chi1arr=np.append(chi1arr, redchisq1)
    chi2arr=np.append(chi2arr, redchisq2)
    chi3arr=np.append(chi3arr, redchisq3)
    chi4arr=np.append(chi4arr, redchisq4) 
    chi5arr=np.append(chi5arr, redchisq5)  
    print 'chisq ratio', chiratio
    print 'justplawchi', just1plaw_chisq
    print 'twocomp chi', model_chisq
    print 'df1 , df2', df1, df2
    ftest1 = stats.f.sf(chiratio1, df1, df3)
    ftest2 = stats.f.sf(chiratio2, df2, df3)
    print 'f-test', ftest1
    print 'f-test2', ftest2
    print 'mjd', qmjd
    ftest_2comp_model= np.append(ftest_2comp_model, ftest1)
    ftest_2comp_model_mplaw= np.append( ftest_2comp_model_mplaw, ftest2)
    onlyplaw_alpha = np.append(onlyplaw_alpha, just1plaw_params['slope'].value)
    sigma_onlyplaw_alpha = np.append(sigma_onlyplaw_alpha, 
    just1plaw_params['slope'].stderr)
#############################################################################
############Plot the best fit 2 component models along with the best powerlaw
    name = '_bestfitmodels.png'
    plot_two_component_model(restnu, normnu, nuspec, normalflux, polarnu, 
    fitflux, sfitflux, mydatamodel, synmodel, bumpmodel, 
    just1plaw_model, plotout, qmjd, name, fitwindow)
    name = '_bestfitmodels_linearpol.png'
    plot_two_component_model(restnu, normnu, nuspec, normalflux, polarnu, 
    fitflux, sfitflux, mydatamodel_2, synmodel_2, bumpmodel_2, just1plaw_model, 
    plotout, qmjd, name, fitwindow)
    name = '_bestfitmodels_quadraticpol.png'
    plot_two_component_model(restnu, normnu, nuspec, normalflux, polarnu, 
    fitflux, sfitflux, mydatamodel_quad, synmodel_quad, bumpmodel_quad, just1plaw_model, 
    plotout, qmjd, name, fitwindow)
    bbbnormalize = np.append(bbbnormalize, 
    twocomp_params['bumpnormal'].value*np.power(normnu, (0.77))*normflux)
    name = '_bestfitmodels_fixedbbb.png'
    plot_two_component_model(restnu, normnu, nuspec, normalflux, polarnu, 
    fitflux, sfitflux, mydatamodel_fixedbbb, synmodel_fixedbbb, 
    bumpmodel_fixedbbb, just1plaw_model, plotout, qmjd, name, fitwindow)
    name = '_bestfitmodels_quadpolwbb.png'
    plot_two_component_model(restnu, normnu, nuspec, normalflux, polarnu,
    fitflux, sfitflux, mydatamodel_fixedbbb, synmodel_fixedbbb, 
    bumpmodel_fixedbbb, just1plaw_model, plotout, qmjd, name, fitwindow)
    
    #   blackbodyfluxmodel_quadpol,  twocomp_BB_params_quadpol, synmodelwbb_quadpol, bumpmodelwbb_quadpol, blackbodymodel_quadpol, synfluxmodel_quadpol, bbbumpfluxmodel_quadpol, mychisquarewbb_quadpol= fit_flux_two_comp_wbb(restnu, polarnu, polarflux, spflux, fitwindow, normflux, plotout, qmjd, oldq, oldu, nuorig, fitflux, twocomp_params_quadpol, normalflux, normnu, sfitflux, twocomp_params, nuspec)
    
    
    
    
    synchrotron_normalize = np.append(synchrotron_normalize, 
    twocomp_params['synnormal'].value*np.power(normnu, 
    twocomp_params['synalpha'].value*(-1))*normflux )
    alpha_array = np.append(alpha_array, plaw_params['alpha'].value)
    alpha_err = np.append(alpha_err, plaw_params['alpha'].stderr)
    linearpol_alpha = np.append(linearpol_alpha, twocomp_params_2['synalpha'].value)
    linearpol_alpha_sigma = np.append(linearpol_alpha_sigma, twocomp_params_2['synalpha'].stderr)
    quadratic_alpha = np.append(quadratic_alpha, twocomp_params_quadpol['synalpha'].value)
    quadratic_alpha_sigma = np.append(quadratic_alpha_sigma, twocomp_params_quadpol['synalpha'].stderr)

print np.median(bbbnormalize)
print len(alpha_array), len(alpha_err), len(mjdarr)
#Plot The Spectral Index for various models######################
###############################################################################
name='twocomp'
plot_alpha(mjdarr, alpha_array, alpha_err, oldplotout, onlyplaw_alpha, 
sigma_onlyplaw_alpha, modified_alpha_array, 
sigma_modified_alpha_array, bbb_integratedflux, name)
name='linearpol'
plot_alpha(mjdarr, linearpol_alpha, linearpol_alpha_sigma, oldplotout, onlyplaw_alpha, 
sigma_onlyplaw_alpha, modified_alpha_array, 
sigma_modified_alpha_array, bbb_integratedflux, name)
name='quadpol'
plot_alpha(mjdarr, quadratic_alpha, quadratic_alpha_sigma, oldplotout, onlyplaw_alpha, 
sigma_onlyplaw_alpha, modified_alpha_array, 
sigma_modified_alpha_array, bbb_integratedflux, name)
###############################################################################
###############################################################################

plot_polflux_sf(mjdarr, ftest_2comp_model, ftest_2comp_model_mplaw, 
oldplotout, 'polarized_flux_models_ftest')
plot_chisquare(mjdarr, chi1arr, chi2arr, chi3arr, chi4arr, blackbodychisquare, chi5arr, oldplotout)
#plot how parameters of interest have changed throughout several years
print len(alpha_array), len(alpha_err), len(mjdarr)

plot_the_final_models(mjdarr, bbb_integratedflux2, synchrotron_integratedflux2, 
total_integratedflux2, oldplotout, bbb_integratedflux_fixedbbb, synchrotron_integratedflux_fixedbbb, total_integratedflux_fixedbbb, total_integratedflux, onlyplaw_alpha, chi1arr, blackbodyflux, temperature_array, bbb_integratedflux, synchrotron_integratedflux, mybbbindex, bbsynmodel, bbbumpmodel, totalfluxwbb, bbb_integratedflux_quad, synchrotron_integratedflux_quad, total_integratedflux_quad,totalfluxwbb_quadpol, bbsynmodel_quadpol, bbbumpmodel_quadpol, blackbodyflux_quadpol)



#plt.plot(mjdarr, blackbodyflux/bbbumpmodel)
#plt.show()
temperature_array = np.array([])
alphaoutput = dirout+'alphaoutput.csv'
with open(alphaoutput, 'wb') as csvfile:
   spamwriter = csv.writer(csvfile, delimiter=',')
   csvfile.write(' MJD, alpha, sigma \n')
   for x in range(len(mjdarr)):
      spamwriter.writerow([mjdarr[x], alpha_array[x], alpha_err[x]])


