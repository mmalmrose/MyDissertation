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
from fit_polarized_flux_plaw_combined import fit_polarized_flux_plaw_combined
from fit_polarized_flux_modified_plaw_combined import fit_polarized_flux_modified_plaw_combined
from fit_flux_two_comp_model import fit_flux_two_comp_model
from fit_flux_two_comp_wbb import fit_flux_two_comp_wbb
from two_comp_modified_plaw import two_comp_modified_plaw
from two_comp_modified_plaw_fixedbbb import two_comp_modified_plaw_fixedbbb
from just_single_plaw import just_single_plaw
from plot_the_final_models import plot_the_final_models
from fit_polarized_flux_modified_plaw_quad import fit_polarized_flux_modified_plaw_quad
from two_comp_modified_plaw_quad import two_comp_modified_plaw_quad
from plot_chisquare_polarized import plot_chisquare_polarized
from find_plaw_expon_cutoff import find_plaw_expon_cutoff
from fit_polarized_flux_plaw_expon_cutoff import fit_polarized_flux_plaw_expon_cutoff
from fit_flux_two_comp_model_expon_cutoff import fit_flux_two_comp_model_expon_cutoff
from fit_flux_two_comp_wbb_exponsyn import fit_flux_two_comp_wbb_exponsyn
from plot_chisquare_better import plot_chisquare_better
from plot_chisquare_polarized_better import plot_chisquare_polarized_better
from plot_alpha_better import plot_alpha_better
#End of importing modules#######################################################

# parse the name of the file that has the input info from the command line
parser = argparse.ArgumentParser()
parser.add_argument("indata",  type=str, 
help="File that Keeps the path to qfiles, ufiles, spectrum, extinction file, redshift, perkins telescope data, and name of source")
args = parser.parse_args()
inputs=np.genfromtxt(args.indata, dtype=None, delimiter=',')
#have the inputs read in, now need to define the output directories.
nameofsource=inputs.item()[5].strip()
dirout = '../outputfiles/'+nameofsource+'/VARBBB/textout/'
plotout = '../outputfiles/'+nameofsource+'/VARBBB/plots/'
asciispectraout = '../outputfiles/'+nameofsource+'/VARBBB/ascii_spectra/'
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
if not os.path.exists(asciispectraout):
    os.makedirs(asciispectraout)
oldplotout = plotout+'/combined/'
#get the name of the emission line template and plot it
tempdir = os.path.dirname(inputs.item()[0].strip())
tempdir = '../'+nameofsource+'/support_files/'
tempfile = tempdir+'template.txt'
#read the template x, y values of the template in, 
#correct it for galactic reddening
redshift = np.float32(inputs.item()[3])
linetemplate= np.genfromtxt(tempfile, dtype=None)
y = correct_extinction(inputs.item()[2].strip(), linetemplate[:,0]*(1.+redshift)
, linetemplate[:,1])
y1 = linetemplate[:,1]
x = linetemplate[:, 0]
c = np.float64(2.99792458e+18) #angstrom/s
xnu = np.float64(c/x)
plot_lines_template(x, y, xnu, plotout)
xobs = x*(1.0+redshift)
y1 = y1 * (xnu**2/2.99792458e+18)
'''
Begin the main loop of the program.  Will read in the q, u, and flux spectra.
subtract the template.  Determine the polarized flux  and fit the models' 
'''

polspectra = np.genfromtxt(inputs.item()[0].strip(), dtype='S150')
fspectra = np.genfromtxt(inputs.item()[1].strip(), dtype ='S150')
size = len(fspectra)

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
quadraticpolwbb_chi = np.array([])
plawchi = np.array([])
linearchi = np.array([])
quadchi = np.array([])
bbsynmodel_modplaw = np.array([])
bbbumpmodel_modplaw = np.array([])
blackbodyflux_modplaw = np.array([])
temperature_array_modplaw = np.array([])
blackbodychisquare_modplaw = np.array([])
totalfluxwbb_modplaw =np.array([])
linearpolwbb_chi = np.array([])
mybbbindex_ec = np.array([])
bbb_integratedflux_ec= np.array([])
synchrotron_integratedflux_ec=np.array([])
total_integratedflux_ec =np.array([])
totalfluxwbb_expon = np.array([])
bbsynmodel_expon = np.array([])
bbbumpmodel_expon = np.array([])
blackbodyflux_expon = np.array([])
exponbb_chi_reduced_arr = np.array([])
expon_chi_reduced_arr = np.array([])
################################################################################
print fspectra
for i in range(size):
# Read the spectrum into an array
#    (qspec, qerr, qend, qheader, qmjd) = readspectrum(qspectra[i])
#    (uspec, uerr, uend, uheader, umjd) = readspectrum(uspectra[i])
    (fspec, ferr, fend, fheader, fmjd) = readspectrum(fspectra[i])
    (polspec, polerr, pend, polheader, pmjd) = readspectrum(polspectra[i])
    mjdarr = np.append(mjdarr, fmjd)
    plotout = oldplotout+'/combined/'+str(fmjd)+'/'
    if not os.path.exists(plotout):
        os.makedirs(plotout)
    #create the array for the wavelength
    wavend = fend*4.0 + 4000.0
    wave = np.linspace(4000, wavend,fend)
    origcorrected = correct_extinction(inputs.item()[2].strip(), wave, fspec)
 #   origcorrected   = fspec
    #subtract the emission lines template
    restwave = wave/(1.0+redshift)
    restnu = c/restwave
    interpfunc=interp1d(x, y, kind='cubic', bounds_error=False)
    newtemplate= interpfunc(restwave)
    thisspec = origcorrected-newtemplate
    specerr = (ferr/c)*restwave**2
    nuspec = np.float64((thisspec/c)*np.power(restwave,2))
    polspec = np.float64((polspec/c)*np.power(restwave,2))
    nuorig = np.float64((origcorrected/c)*np.power(restwave,2))
    plot_spectra_orig(origcorrected, thisspec, nuorig, nuspec, restwave, restnu
    , plotout, fmjd)
    if nuspec.shape != fspec.shape:
        nuspec= rebin(nuspec, fspec.shape)
    
    #correct polarization for statistical bias (Wardle and Kronberg, 1974)
    #p = np.sqrt(qspec**2 + uspec**2)
    # bin q and u in order to determine polarization and the direction of polarization
    boxcarwindow = 15
   # p, sp, thetp, sthetp, qspec, sqspec, uspec, suspec, oldq, oldu = defpol(
    #qspec, uspec, boxcarwindow)  
    #print len(nuorig), 'len nuorig'
    #lnorig = len(nuspec)
    #print lnorig
    #otherpolspec, otherpolerr = bin_array(nuorig*np.sqrt(oldu**2 + oldq**2), lnorig/boxcarwindow)  
    #bin the q, u, and flux arrays
    numberofbins = len(nuorig)/boxcarwindow
    polarflux, spflux = bin_array(nuorig, numberofbins)
    polarnu, spolarnu = bin_array(restnu, numberofbins)    
    fitflux, sfitflux = bin_array(nuspec, numberofbins)
    sfitflux = sfitflux * np.sqrt(boxcarwindow)
    polarflux,spflux = bin_array(polspec, numberofbins)
    p = polarflux/fitflux
    sp = sfitflux/100.
    #spflux = np.sqrt((polarflux*sp)**2 + (spflux*p)**2) * np.sqrt(boxcarwindow)*3
    polarwave = c/polarnu
    #plot_q_u_p(qspec, sqspec, uspec, suspec, p, sp, thetp, sthetp, polarnu, 
    #polarwave, plotout, qmjd, oldq, oldu, restnu)
    normflux = 10.0**np.fix(np.log10(np.mean(polarflux)))
    normnu = 10.0**np.fix(np.log10(np.mean(polarnu)))
################################################################################
################Print the spectrum into an ascii file###########################
#    spectrumfile = asciispectraout+str(qmjd)+'_spectrum.csv'
#    print spectrumfile, 'spectrumfile'
#    with open(spectrumfile, 'wb') as csvfile:
#       spamwriter = csv.writer(csvfile, delimiter=',')
#       csvfile.write('nu, flux, sigma \n')
#       for aa in range(len(restnu)):
#            spamwriter.writerow([restnu[aa], nuspec[aa], specerr[aa]])
#################### Fit the polarized flux with a power-law####################
##### use the lmfit package to determine alpha##################################
#############DEFINE THE FREQUENCIES FOR WHICH THE FIT WILL BE PERFORMED#########
###############################################################################
    fitwindow = np.array([])
    fitwindowa = np.where((polarnu > 6.2e+14) &( polarnu < 6.85e+14)) 
    #fitwindowa = np.where((polarnu > 6.2e+14) &( polarnu <10.2e+14)) 
    fitwindowb = np.where((polarnu > 9.08e+14) & (polarnu < 10.2e+14))
    fitwindow = np.append(fitwindowa, fitwindowb).astype(int)
    #fitwindow = np.append(fitwindow, fitwindowb).astype(int)
    qmjd = fmjd
################################################################################
#######################Fit the polarized flux with a power law##################
    initialnormal = np.float64(np.median(polarflux)*np.median(polarnu))/normflux
    plaw_params, output, model = fit_polarized_flux_plaw_combined(polarnu, polarflux, 
    normflux, spflux, initialnormal, fitwindow, qmjd, plotout, restnu, nuorig, polspec)
    #print plaw_params, 'plaw params'

########### Fit the polarized flux with a modified power-law###################
##########  F = (A * nu + B) * Norm * nu^(-alpha)##############################
########## Start with fitting the polarization with P = (A nu + B)##############
    linepol_params, linepol_output, linepol_model, modified_plaw_params,modified_plaw_model =fit_polarized_flux_modified_plaw_combined(restnu, polarnu, 
    polarflux, spflux, fitwindow, normflux, plotout, qmjd, nuorig, p, sp, plaw_params, polspec)
    #print linepol_params, 'linepol_params'
########### Fit the polarized flux with a modified power-law####################
########### of the form F = (A * nu^2 + B*nu + C) * NORM * nu^(-alpha)##########
    quadpol_params, quadpol_output, quadpol_model, modified_plaw_params_quad, modified_plaw_model_quad = fit_polarized_flux_modified_plaw_quad(restnu, polarnu, 
    polarflux, spflux, fitwindow, normflux, plotout, qmjd, oldq, oldu, 
    nuorig, p, sp, plaw_params)
    #print quadpol_params, 'quadpol_params'
################################################################################

################################################################################
#######################Fit the polarized flux with a power law##################
####################with an exponential cutoff##################################
    initialnormal = np.float64(np.median(polarflux)*np.median(polarnu))/normflux
    plaw_params_expon, output_expon, model_expon = fit_polarized_flux_plaw_expon_cutoff(polarnu, polarflux, 
    normflux, spflux, initialnormal, fitwindow, qmjd, plotout, restnu, oldq,
     oldu, nuorig)
    print plaw_params_expon, 'plaw params exponential cutoff'


################################################################################
###############################################################################
# Perform an F-test on the two models for the polarized flux###################
    chisquareplaw = chi_square(model, polarflux, spflux)
    chisquarelinear = chi_square(modified_plaw_model, polarflux, spflux)
    chisquarequad = chi_square(modified_plaw_model_quad, polarflux, spflux)
    chiratio1 = chisquareplaw/chisquarelinear
    chiratio2 = chisquareplaw/chisquarequad
    dof1 = len(model) - 2
    dof2 = len(model) - 3
    dof3 = len(model) - 4
    red_chisquareplaw = chisquareplaw/dof1
    red_chisquarelinear = chisquarelinear/dof2
    red_chisquarequad = chisquarequad/dof3
    sf_polarized_flux = stats.f.sf(chiratio1, dof1, dof2)
    sf_polarized_flux2 = stats.f.sf(chiratio2, dof1, dof3)
    polarized_flux_sf = np.append(polarized_flux_sf, sf_polarized_flux)
    polarized_flux_sf2 = np.append(polarized_flux_sf2, sf_polarized_flux2)
    plawchi = np.append(plawchi, red_chisquareplaw)
    linearchi = np.append(linearchi, red_chisquarelinear)
    quadchi = np.append(quadchi, red_chisquarequad)
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


####################Perform the fit for the model###############################
#############with the cutoff exponential synchrotron############################
    modelflux_ec, modelsyn_ec, synmodel_ec, mymodel_ec, bumpmodel_ec, model_chisq_ec,  model_reduced_chi_ec, twocomp_params_ec, model_for_spectrum_ec, mydatamodel_ec, df1, modelbump_ec =fit_flux_two_comp_model_expon_cutoff(restnu, polarnu, polarflux, spflux, fitwindow, normflux, plotout, qmjd, oldq, oldu, nuorig, fitflux, plaw_params_expon, normalflux, normnu, sfitflux)
    mybbbindex_ec = np.append(mybbbindex, twocomp_params_ec['bumpindex'].value) 
    bbb_integratedflux_ec= np.append(bbb_integratedflux_ec, modelbump_ec)
    synchrotron_integratedflux_ec=np.append(synchrotron_integratedflux_ec, modelsyn_ec)
    total_integratedflux_ec = np.append(total_integratedflux_ec, modelflux_ec)
    print twocomp_params_ec, 'twocomp_params_ec'
    exponchisquared = chi_square(mydatamodel_ec[fitwindow], fitflux[fitwindow], 
    sfitflux[fitwindow])
    exponchisquared_reduced = exponchisquared/(len(fitwindow) - 4)
    expon_chi_reduced_arr = np.append(expon_chi_reduced_arr, exponchisquared_reduced)
 ###############################################################################


##

###############################################################################
################Perform the two-component fit with the alpha obtained########
################# By fitting the polarized flux with the modified Plaw#######
    twocomp_params_2, model_for_spectrum_2, mymodel_2, mydatamodel_2, synmodel_2,bumpmodel_2, modelflux_2, modelsyn_2, modelbump_2 = two_comp_modified_plaw(fitflux, sfitflux, normalflux, polarnu, normnu, plaw_params, twocomp_params, restnu, modified_plaw_params, fitwindow)   
    bbb_integratedflux2= np.append(bbb_integratedflux2, modelbump_2)
    synchrotron_integratedflux2=np.append(synchrotron_integratedflux2, 
    modelsyn_2)
    total_integratedflux2 = np.append(total_integratedflux2, modelflux_2)

###############################################################################
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
################################################################################
#####################Fit Just  a power-law######################################
################################################################################
    just1plaw_model, just1plaw_output, just1plaw_params=just_single_plaw(fitflux, sfitflux, normalflux, polarnu, normnu, plaw_params, twocomp_params, restnu, modified_plaw_params, fitwindow)
    just1plaw_chisq = chi_square(just1plaw_model[fitwindow], fitflux[fitwindow],
    sfitflux[fitwindow])
    just1plaw_reduced_chi = just1plaw_chisq/(np.float(len(fitwindow)) - 2)


################################################################################
 ###################Perform the Two-Component fit with a BB on top##############
    #perform the fit for the two component model.
    #first select the windows for the fit
    name='blackbody'
    blackbodyfluxmodel,  twocomp_BB_params, synmodelwbb, bumpmodelwbb, blackbodymodel, synfluxmodel, bbbumpfluxmodel, mychisquarewbb= fit_flux_two_comp_wbb(restnu, polarnu, polarflux, spflux, fitwindow, normflux, plotout, qmjd, oldq, oldu, nuorig, fitflux, plaw_params, normalflux, normnu, sfitflux, twocomp_params, nuspec,name, just1plaw_model)
    totalfluxwbb = np.append(totalfluxwbb, synfluxmodel+bbbumpfluxmodel+blackbodyfluxmodel)
    bbsynmodel = np.append(bbsynmodel, synfluxmodel)
    bbbumpmodel = np.append(bbbumpmodel, bbbumpfluxmodel)  
    blackbodyflux = np.append(blackbodyflux, blackbodyfluxmodel)
    modelwblackbody = blackbodymodel+synmodelwbb+bumpmodelwbb
    mychisquarewbb =chi_square(modelwblackbody[fitwindow], fitflux[fitwindow], sfitflux[fitwindow])
    temperature_array = np.append(temperature_array, twocomp_BB_params['temperature'].value)
    blackbodychisquare = np.append(blackbodychisquare, mychisquarewbb)
###############################################################################
###############################################################################
 ################Perform the two-component fit with the alpha obtained########
 ################# By fitting the polarized flux with the modified Plaw#######
 ################# and adding a blackbody ##########################
    name='blackbody_modifiedplaw'
    blackbodyfluxmodel_modplaw,  twocomp_BB_params_modplaw, synmodelwbb_modplaw, bumpmodelwbb_modplaw, blackbodymodel_modplaw, bbsynfluxmodel_modplaw, bbbumpfluxmodel_modplaw, mychisquarewbb_modplaw= fit_flux_two_comp_wbb(restnu, polarnu, polarflux, spflux, fitwindow, normflux, plotout, qmjd, oldq, oldu, nuorig, fitflux, plaw_params, normalflux, normnu, sfitflux, twocomp_params_2, nuspec,name, just1plaw_model)
    totalmodelherenow = blackbodymodel_modplaw+synmodelwbb_modplaw+bumpmodelwbb_modplaw
    totalfluxwbb_modplaw = np.append(totalfluxwbb_modplaw, bbsynfluxmodel_modplaw+bbbumpfluxmodel_modplaw+blackbodyfluxmodel_modplaw)
    bbsynmodel_modplaw = np.append(bbsynmodel_modplaw, bbsynfluxmodel_modplaw)
    bbbumpmodel_modplaw = np.append(bbbumpmodel_modplaw, bbbumpfluxmodel_modplaw)  
    blackbodyflux_modplaw = np.append(blackbodyflux_modplaw, blackbodyfluxmodel_modplaw)
    temperature_array_modplaw = np.append(temperature_array_modplaw, twocomp_BB_params_modplaw['temperature'].value)
    blackbodychisquare_modplaw = np.append(blackbodychisquare_modplaw, mychisquarewbb_modplaw)
    chisquarelinepolwbb = chi_square(totalmodelherenow[fitwindow], fitflux[fitwindow], sfitflux[fitwindow])
    degreesoffreedom = len(fitwindow)-4
    redchisquare_linearpolwbb = chisquarelinepolwbb/degreesoffreedom
 ###############################################################################
 ###############################################################################
 ################Perform the two-component fit with the alpha obtained########
 ################# By fitting the polarized flux with the modified Plaw#######    
 ################# with a quadratic term ######################################   
    twocomp_params_quadpol, quad_model_for_spectrum, mymodel_quad, mydatamodel_quad, synmodel_quad, bumpmodel_quad, modelflux_quad, modelsyn_quad, modelbump_quad = two_comp_modified_plaw_quad(fitflux, sfitflux, normalflux, polarnu, normnu, plaw_params, quadpol_params, restnu, modified_plaw_params_quad, fitwindow)
    bbb_integratedflux_quad= np.append(bbb_integratedflux_quad, modelbump_quad)
    synchrotron_integratedflux_quad=np.append(synchrotron_integratedflux_quad, modelsyn_quad)
    total_integratedflux_quad = np.append(total_integratedflux_quad, modelflux_quad)
################################################################################

################################################################################
#####################Try Quadratic Polarization with a blackbody################
##################### Component on top #########################################
    twocomp_params_quadpol.add('alpha', value = twocomp_params_quadpol['synalpha'].value, vary=False)
    name='quadraticpol'
    blackbodyfluxmodel_quadpol,  twocomp_BB_params_quadpol, synmodelwbb_quadpol, bumpmodelwbb_quadpol, blackbodymodel_quadpol, synfluxmodel_quadpol, bbbumpfluxmodel_quadpol, mychisquarewbb_quadpol = fit_flux_two_comp_wbb(restnu, polarnu, polarflux, spflux, fitwindow, normflux, plotout, qmjd, oldq, oldu, nuorig, fitflux, twocomp_params_quadpol, normalflux, normnu, sfitflux, twocomp_params, nuspec, name, just1plaw_model)
    totalmodelhere = bumpmodelwbb_quadpol + blackbodymodel_quadpol + synmodelwbb_quadpol 
    totalfluxwbb_quadpol = np.append(totalfluxwbb_quadpol, synfluxmodel_quadpol + bbbumpfluxmodel_quadpol + blackbodyfluxmodel_quadpol)
    bbsynmodel_quadpol = np.append(bbsynmodel_quadpol, synfluxmodel_quadpol)
    bbbumpmodel_quadpol = np.append(bbbumpmodel_quadpol, bbbumpfluxmodel_quadpol)
    blackbodyflux_quadpol = np.append(blackbodyflux_quadpol, blackbodyfluxmodel_quadpol)
    #print len(totalmodelhere)
    chisquare_quadpolwbb = chi_square(totalmodelhere[fitwindow], fitflux[fitwindow], sfitflux[fitwindow])
    degreesoffreedom = len(fitwindow)-5
    redchisquare_quadpolwbb = chisquare_quadpolwbb/degreesoffreedom
    
    
    
####################Perform the fit for the model###############################
#############with the cutoff exponential synchrotron############################
############# And a blackbody ################################################    
    name='expon_synchrotron_wbb'
    blackbodyfluxmodel_expon,  twocomp_BB_params_expon, synmodelwbb_expon, bumpmodelwbb_expon, blackbodymodel_expon, synfluxmodel_expon, bbbumpfluxmodel_expon, mychisquarewbb_expon = fit_flux_two_comp_wbb_exponsyn(restnu, polarnu, polarflux, spflux, fitwindow, normflux, plotout, qmjd, oldq, oldu, nuorig, fitflux, twocomp_params_ec, normalflux, normnu, sfitflux, twocomp_params, nuspec, name, just1plaw_model)
    totalmodelhere = bumpmodelwbb_expon + blackbodymodel_expon + synmodelwbb_expon 
    totalfluxwbb_expon = np.append(totalfluxwbb_expon, synfluxmodel_expon + bbbumpfluxmodel_expon + blackbodyfluxmodel_expon)
    bbsynmodel_expon = np.append(bbsynmodel_expon, synfluxmodel_expon)
    bbbumpmodel_expon = np.append(bbbumpmodel_expon, bbbumpfluxmodel_expon)
    blackbodyflux_expon = np.append(blackbodyflux_expon, blackbodyfluxmodel_expon)
    #print len(totalmodelhere)
    chisquare_exponwbb = chi_square(totalmodelhere[fitwindow], fitflux[fitwindow], sfitflux[fitwindow])
    chisquare_exponwbb_reduced = chisquare_exponwbb/(len(fitwindow) - 5)
    exponbb_chi_reduced_arr = np.append(exponbb_chi_reduced_arr, chisquare_exponwbb_reduced)
###############################################################################
    #############Perform a fit of a power-law to the spectrum##################
    ########### Do an F-Test with this fit and each of the 2 component models###########
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
    quadraticpolwbb_chi = np.append(quadraticpolwbb_chi, redchisquare_quadpolwbb)
    linearpolwbb_chi = np.append( linearpolwbb_chi, redchisquare_linearpolwbb)
    ftest1 = stats.f.sf(chiratio1, df1, df3)
    ftest2 = stats.f.sf(chiratio2, df2, df3)
    print 'f-test', ftest1
    print 'f-test2', ftest2
    print 'mjd', qmjd
    ftest_2comp_model= np.append(ftest_2comp_model, ftest1)
    ftest_2comp_model_mplaw= np.append( ftest_2comp_model_mplaw, ftest2)
    onlyplaw_alpha = np.append(onlyplaw_alpha, just1plaw_params['slope'].value*(-1.0))
    sigma_onlyplaw_alpha = np.append(sigma_onlyplaw_alpha, 
    just1plaw_params['slope'].stderr)
################################################################################
############Plot the best fit 2 component models along with the best powerlaw
    name = '_bestfitmodels.png'
    plot_two_component_model(restnu, normnu, nuspec, normalflux, polarnu, 
    fitflux, sfitflux, mydatamodel, synmodel, bumpmodel, 
    just1plaw_model, plotout, qmjd, name, fitwindow, y1)
    name = '_bestfitmodels_linearpol.png'
    plot_two_component_model(restnu, normnu, nuspec, normalflux, polarnu, 
    fitflux, sfitflux, mydatamodel_2, synmodel_2, bumpmodel_2, just1plaw_model, 
    plotout, qmjd, name, fitwindow, y1)
    name = '_bestfitmodels_quadraticpol.png'
    plot_two_component_model(restnu, normnu, nuspec, normalflux, polarnu, 
    fitflux, sfitflux, mydatamodel_quad, synmodel_quad, bumpmodel_quad, just1plaw_model, 
    plotout, qmjd, name, fitwindow,y1)
    bbbnormalize = np.append(bbbnormalize, 
    twocomp_params['bumpnormal'].value*np.power(normnu, (0.77))*normflux)
    name = '_bestfitmodels_fixedbbb.png'
    plot_two_component_model(restnu, normnu, nuspec, normalflux, polarnu, 
    fitflux, sfitflux, mydatamodel_fixedbbb, synmodel_fixedbbb, 
    bumpmodel_fixedbbb, just1plaw_model, plotout, qmjd, name, fitwindow,y1)
    name = '_exponential_cutoff.png'
    plot_two_component_model(restnu, normnu, nuspec, normalflux, polarnu, 
    fitflux, sfitflux, mydatamodel_ec, synmodel_ec, 
    bumpmodel_ec, just1plaw_model, plotout, qmjd, name, fitwindow,y1)
    synchrotron_normalize = np.append(synchrotron_normalize, 
    twocomp_params['synnormal'].value*np.power(normnu, 
    twocomp_params['synalpha'].value*(-1))*normflux )
    alpha_array = np.append(alpha_array, plaw_params['alpha'].value)
    alpha_err = np.append(alpha_err, plaw_params['alpha'].stderr)
    linearpol_alpha = np.append(linearpol_alpha, twocomp_params_2['synalpha'].value)
    linearpol_alpha_sigma = np.append(linearpol_alpha_sigma, twocomp_params_2['synalpha'].stderr)
    quadratic_alpha = np.append(quadratic_alpha, twocomp_params_quadpol['synalpha'].value)
    quadratic_alpha_sigma = np.append(quadratic_alpha_sigma, twocomp_params_quadpol['synalpha'].stderr)
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
name='twocomp_w_ebar'
plot_alpha_better(mjdarr, alpha_array, alpha_err, oldplotout, onlyplaw_alpha, sigma_onlyplaw_alpha, name)
###############################################################################
###############################################################################
#Plot the survival functions for the various polarized flux models
plot_polflux_sf(mjdarr, polarized_flux_sf, polarized_flux_sf2, oldplotout, 'polarized_flux_models_ftest')
#plot the reduced chisquare for the various polarized flux 
#plot_chisquare_polarized(mjdarr, plawchi, linearchi, quadchi,oldplotout) #linearchi, quadchi
plot_chisquare_polarized_better(mjdarr, plawchi, 'Polarized-Flux Power-Law', 'polarized_flux_plaw.png',oldplotout)
plot_chisquare_polarized_better(mjdarr, linearchi, 'Polarized-Flux Power-Law w Linear-Term', 'polarized_flux_linear.png',oldplotout)
plot_chisquare_polarized_better(mjdarr, plawchi, 'Polarized-Flux Power-Law w Quadratic-Term', 'polarized_flux_quad.png',oldplotout)
#plot_polflux_sf(mjdarr, ftest_2comp_model, ftest_2comp_model_mplaw, 
#oldplotout, 'polarized_flux_models_ftest')
#plot_chisquare(mjdarr, chi1arr, chi2arr, chi3arr, chi4arr, blackbodychisquare, chi5arr, quadraticpolwbb_chi,  linearpolwbb_chi, oldplotout)
plot_chisquare_better(mjdarr, exponbb_chi_reduced_arr,'Exponential Cutoff Polarization With Blackbody', 'chisquare_exponsynwbb.png', oldplotout)
plot_chisquare_better(mjdarr, expon_chi_reduced_arr,'Exponential Polarization', 'chisquare_exponsyn.png', oldplotout)
plot_chisquare_better(mjdarr, chi1arr,'Two-Component Model', 'chisquare_2compmodel.png', oldplotout)
plot_chisquare_better(mjdarr, chi2arr,'Linear Polarization', 'chisquare_linearpol.png', oldplotout)
plot_chisquare_better(mjdarr, chi3arr,'Just Power-Law', 'chisquare_justplaw.png', oldplotout)
plot_chisquare_better(mjdarr, chi4arr,'Fixed-BBB', 'chisquare_fixedBBB.png', oldplotout)
plot_chisquare_better(mjdarr, chi5arr,'Quadratic Polarization', 'chisquare_quadraticpol.png', oldplotout)
plot_chisquare_better(mjdarr, blackbodychisquare,'Two-Component Model With Black-Body', 'chisquare_2compmodelwbb.png', oldplotout)
plot_chisquare_better(mjdarr, quadraticpolwbb_chi,'Quadratic Polarization With Black-Body', 'chisquare_quadpolwbb.png', oldplotout)
plot_chisquare_better(mjdarr, linearpolwbb_chi,'Linear Polarization With Black-Body', 'chisquare_linearpolarization.png', oldplotout)

#plot how parameters of interest have changed throughout several years
plot_the_final_models(mjdarr, bbb_integratedflux2, synchrotron_integratedflux2, 
total_integratedflux2, oldplotout, bbb_integratedflux_fixedbbb, synchrotron_integratedflux_fixedbbb, total_integratedflux_fixedbbb, total_integratedflux, onlyplaw_alpha, chi1arr, blackbodyflux, temperature_array, bbb_integratedflux, synchrotron_integratedflux, mybbbindex, bbsynmodel, bbbumpmodel, totalfluxwbb, bbb_integratedflux_quad, synchrotron_integratedflux_quad, total_integratedflux_quad,totalfluxwbb_quadpol, bbsynmodel_quadpol, bbbumpmodel_quadpol, blackbodyflux_quadpol,totalfluxwbb_modplaw, bbsynmodel_modplaw, bbbumpmodel_modplaw, blackbodyflux_modplaw, bbb_integratedflux_ec, synchrotron_integratedflux_ec, total_integratedflux_ec,totalfluxwbb_expon, bbsynmodel_expon, bbbumpmodel_expon, blackbodyflux_expon)

alphaoutput = dirout+'alphaoutput.csv'
with open(alphaoutput, 'wb') as csvfile:
   spamwriter = csv.writer(csvfile, delimiter=',')
   csvfile.write(' MJD, alpha, sigma \n')
   for xx in range(len(mjdarr)):
      spamwriter.writerow([mjdarr[xx], alpha_array[xx], alpha_err[xx], onlyplaw_alpha[xx], 
sigma_onlyplaw_alpha[xx]])


