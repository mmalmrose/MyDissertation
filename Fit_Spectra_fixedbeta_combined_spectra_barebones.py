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
from scipy import interpolate
import lmfit
from lmfit import  Model, minimize, Parameters
from scipy import stats
from scipy.stats import ks_2samp
from scipy.stats import pearsonr
#import functions I've written##################################################
from correct_extinction import correct_extinction
from readspectrum_combinedspec import readspectrum_combinedspec
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
from fit_polarized_flux_modified_plaw_lin import fit_polarized_flux_modified_plaw_lin
from fit_flux_two_comp_model_mc import fit_flux_two_comp_model_mc
from fit_flux_two_comp_wbb import fit_flux_two_comp_wbb
from two_comp_modified_plaw_mc import two_comp_modified_plaw_mc
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
from plot_models_mc import plot_models_mc
from fit_flux_two_comp_wbb_mc import fit_flux_two_comp_wbb_mc
from plot_modelswbb_mc import plot_modelswbb_mc
from plot_alphaVchi import plot_alphaVchi
from plot_alpha_better_chi import plot_alpha_better_chi
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
plotout = '../outputfiles/'+nameofsource+'/VARBBB/plots/combined/'
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
oldplotout = plotout
#get the name of the emission line template and plot it
tempdir = os.path.dirname(inputs.item()[0].strip())
tempdir = '../'+nameofsource+'/support_files/'
tempfile = tempdir+'template.txt'
bllac_galaxy_flux ='../BLLac/support_files/bllac_galaxy_flux.txt'
#print tempfile
#read the template x, y values of the template in, 
#correct it for galactic reddening
redshift = np.float32(inputs.item()[3])
linetemplate= np.genfromtxt(tempfile, dtype=[('lambda', 'float'), ('flux', 'float')], delimiter=',')
bllac_galflux = np.genfromtxt(bllac_galaxy_flux, dtype=[('nu','float'), ('flux', 'float')], delimiter=',')
#plt.plot(bllac_galflux['nu'], bllac_galflux['flux'])
#plt.show()
bad = np.where(linetemplate['flux'] <= 0)
linetemplate['flux'][bad] = np.std(linetemplate['flux'])
y = correct_extinction(inputs.item()[2].strip(), linetemplate['lambda']*(1.+redshift), linetemplate['flux'])
y1 = linetemplate['flux']
#y = y1
x = linetemplate['lambda']
y[bad] = 0
y1[bad]=0
c = np.float64(2.99792458e+18) #angstrom/s
xnu = np.float64(c/x)
#plot_lines_template(x, y, xnu, plotout)
xobs = x*(1.0+redshift)
y1 = y1 * (xnu**2/2.99792458e+18)
'''
Begin the main loop of the program.  Will read in the polarized and flux spectra.
subtract the template and fit the models' 
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
sigfluxarr = np.array([])
sigsynarr= np.array([])
sigbumparr = np.array([])
sigfluxarr_wbb = np.array([])
sigbbbumpfluxmodel_wbb = np.array([])
sigblackbodyfluxmodel_wbb = np.array([])
sigsynfluxmodel_wbb = np.array([])
sigbbb_integratedflux2= np.array([])
sigsynchrotron_integratedflux2=np.array([])
sigtotal_integratedflux2 = np.array([])
sigbbsynmodel_modplaw =  np.array([])
sigbbbumpmodel_modplaw =  np.array([])
sigtotalfluxwbb_modplaw =  np.array([])
sigblackbodyflux_modplaw = np.array([])
bb_normalarray = np.array([])
modalpha_array = np.array([])
minusbbbalpha = np.array([])
sigmaminusbbbalpha= np.array([])
minusfixedbbbalpha = np.array([])
sigmaminusfixedbbbalpha= np.array([])
lambdadepalpha= np.array([])
siglambdadepalpha = np.array([])
singleplawchisquare = np.array([])
chisquare_minus_fixedbbb = np.array([])
chisquare_two_comp_model = np.array([])
total_polarized_flux = np.array([])
observebbfrac= np.array([])
polarization_slopes = np.array([])
poldiff_arr = np.array([])
J_proj = np.array([])
H_proj = np.array([])
K_proj = np.array([])
J_proj_wbb = np.array([])
H_proj_wbb = np.array([])
K_proj_wbb = np.array([])
temperature_error =np.array([])
################################################################################
(fspec, ferr, fend, fheader, fmjd) = readspectrum_combinedspec(fspectra[0])
#create the array for the wavelength
wavend = fend*4.0 + 4000.0
wave = np.linspace(4000.0, wavend,fend)
restwave = wave/(1.0+redshift)
restnu = c/restwave
nuspec = np.float64((fspec/c)*np.power(restwave,2))
normalflux = gom(nuspec)
normallambdaflux = gom(fspec)

#Read in J, H, and K_s response files in order to project the synchrotron component into the IR
mimir_H_file = 'Barr_H_BU_MAN102_trans.txt'
mimir_J_file = 'Barr_J_BU_MAN236_trans.txt'
mimir_K_file = 'Barr_Ks_BU_MAN301_trans.txt'

J_resp= np.genfromtxt(mimir_J_file, skip_header=0, delimiter=',', dtype=[('Jlam', 'f'), ('Jres' , 'f')])
H_resp= np.genfromtxt(mimir_H_file, skip_header=0, delimiter=',', dtype=[('Hlam', 'f'), ('Hres' , 'f')])
K_resp= np.genfromtxt(mimir_K_file, skip_header=0, delimiter=',', dtype=[('Klam', 'f'), ('Kres' , 'f')])

rest_jlam = 10000.0*J_resp['Jlam']/(1.+redshift)
rest_hlam = 10000.0*H_resp['Hlam']/(1.+redshift)
rest_klam = 10000.0*K_resp['Klam']/(1.+redshift)
rest_jnu = c/rest_jlam
rest_hnu = c/ rest_hlam
rest_knu = c/rest_klam
#plt.plot(rest_jnu, J_resp['Jres'])
#plt.plot(rest_hnu, H_resp['Hres'])
#plt.plot(rest_knu, K_resp['Kres'])
#plt.show()
for i in range(size):
# Read the spectrum into an array
    #(qspec, qerr, qend, qheader, qmjd) = readspectrum(qspectra[i])
    #(uspec, uerr, uend, uheader, umjd) = readspectrum(uspectra[i])
    (fspec, ferr, fend, fheader, fmjd) = readspectrum_combinedspec(fspectra[i])
    if fspec.shape != wave.shape:
        wave = rebin(wave, fspec.shape)
        restwave = rebin(restwave, fspec.shape)
        restnu = rebin(restnu, fspec.shape)
    print np.median(ferr/fspec)
    noisetosignal= np.median(ferr/fspec)
    qmjd = np.float32(fmjd)

    #print ferr/fspec, 'ferr/fspec'
    if qmjd > 2450000:
        qmjd = qmjd-2450000
    (polspec, polerr, pend, polheader, pmjd) = readspectrum_combinedspec(polspectra[i])
    mjdarr = np.append(mjdarr, qmjd)
    plotout = oldplotout+str(fmjd)+'/'
    if not os.path.exists(plotout):
        os.makedirs(plotout)
    origcorrected = correct_extinction(inputs.item()[2].strip(), wave, fspec)

    interpfunc=interp1d(x, y, kind='cubic', bounds_error=False)
    newtemplate= interpfunc(restwave)
    #get the template error so that we can add it in to ferr
    temperror = newtemplate
    insertzeros = np.zeros(5)
    temperror=np.insert(temperror, 0 , insertzeros)
    temperror = temperror[0:len(newtemplate)]
    temperror = newtemplate-temperror
    temperrorbar = np.nanstd(temperror)
    print temperrorbar
    #temperror[0:] = temperrorbar
    temperror[0:] =0
    thisspec = origcorrected- newtemplate
    nuspec = np.float64((thisspec/c)*np.power(restwave,2))
    print np.mean(ferr), np.mean(temperror), 'ferr, temperror'
    specerr = (np.sqrt(ferr**2 + temperror**2)/fspec)*nuspec
    specerr = nuspec*0.1
    
    polspec = np.float64((polspec/c)*np.power(restwave,2))
    nuorig = np.float64((origcorrected/c)*np.power(restwave,2))
    #for BL Lac, remove the galaxy flux
    print restnu
    newgalflux=0
    from scipy import interpolate as interpolate 
    newgalflux = interpolate.griddata(bllac_galflux['nu'], bllac_galflux['flux'], restnu, method='cubic')
    #plt.plot(restnu, nuspec)
    nuspec = nuspec #- newgalflux
    #plt.plot(restnu, nuspec)
    #plt.plot(restnu, newgalflux)
    #plt.show()
    newgalflux = 0
    print len(origcorrected), len(thisspec), len(nuorig), len(nuspec), len(restwave), len(restnu)
    #plt.plot(nuspec, nuspec)
    #plt.show()
    plot_spectra_orig(origcorrected, thisspec, nuorig, nuspec, restwave, restnu
    , plotout, qmjd,normallambdaflux, normalflux)
    if nuspec.shape != fspec.shape:
        nuspec= rebin(nuspec, fspec.shape)
    uspec = (polspec/nuorig)/np.sqrt(2)
    qspec = (polspec/nuorig)/np.sqrt(2)
    oldu= uspec
    oldq= qspec 
    boxcarwindow = 15

    numberofbins = len(nuorig)/boxcarwindow
    polarflux, spflux = bin_array(nuorig, numberofbins)
    polarnu, spolarnu = bin_array(restnu, numberofbins)   
    fitflux, sfitflux = bin_array(nuspec, numberofbins)
    binorig, sbinorig = bin_array(nuspec, numberofbins)
    sfitflux = fitflux * noisetosignal/np.sqrt(boxcarwindow)
    #sfitflux= sfitflux/np.sqrt(boxcarwindow)
   #Get the new error based on the error in the measured magnitude
    sigmamag = np.sqrt((0.03**2) + (0.04**2))
    NtoF = (np.log(10)/2.5)*sigmamag
    NtoF = 0.03
    #sfitflux = sfitflux + fitflux*NtoF #* np.sqrt(boxcarwindow)
    sfitflux = fitflux*0.02
    polarflux,spflux = bin_array(polspec, numberofbins)
    unpolflux = fitflux - polarflux
    p = polarflux/binorig
    pmeanearly = 2.0*np.mean(p[(len(p)-7):])
    print pmeanearly
    Fsyn = fitflux*p/pmeanearly
   # plt.plot(polarnu,Fsyn)
    #plt.plot(polarnu, fitflux)
    #plt.plot(polarnu, fitflux-Fsyn)
    #plt.show()
    sp = p/100.0
    polarwave = c/polarnu
    normpflux = 10.0**np.fix(np.log10(np.mean(polarflux)))
    normflux = normalflux
    normnu = 10.0**np.fix(np.log10(np.mean(polarnu)))
    total_polarized_flux = np.append(total_polarized_flux, np.trapz(polarflux, polarnu)*(-1))
   


#############DEFINE THE FREQUENCIES FOR WHICH THE FIT WILL BE PERFORMED#########
###############################################################################
    fitwindow = np.array([])
    #fitwindowa = np.where((polarnu/(1.0+redshift) > 6.2e+14) &( polarnu/(1.0+redshift) < 6.85e+14)) 
    #fitwindow = np.where((polarnu/(1.0+redshift) > 6.8e+14) &( polarnu/(1.0+redshift) <9.8e+14)) 
    fitwindow = np.where(((polarnu/ (1.0+redshift)) > 4.1e+14) &( polarnu/(1.0+redshift) <6.5e+14)) 

    #fitwindowb = np.where((polarnu > 9.08e+14) & (polarnu < 10.2e+14))
    #fitwindow = np.append(fitwindowa, fitwindowb).astype(int)
    #fitwindow = np.append(fitwindowa).astype(int)
    
####################Perform a linear fit to the linear polarization %####################
######## With time.  The change Should indicate an increasing  importance########
########## of either the synchrotron or the BBB component. ###################3
    lineforpol = Parameters()
    lineforpol.add('intercept', value=0)
    lineforpol.add('slope', value=np.float64(1.0/4.0))
    lineforpol_output = minimize(find_line_resid, lineforpol, 
        args=(polarnu, p, sp))
    print lineforpol['slope'].value ,'linearpol params'
    polarization_slopes = np.append(polarization_slopes, lineforpol['slope'].value)
    #plt.plot(polarnu, p)
    #plt.plot(polarnu, polarnu*lineforpol['slope'].value + lineforpol['intercept'].value)
    deltap = lineforpol['slope'].value * (np.max(polarnu) - np.min(polarnu))
    print deltap, 'deltap'
    poldiff_arr = np.append(poldiff_arr, deltap)
    #plt.show()
#################### Fit the polarized flux with a power-law####################
##### use the lmfit package to determine alpha##################################

    #plt.plot(polarnu, polarflux)
    #plt.show()
################################################################################
#######################Fit the polarized flux with a power law##################
    initialnormal = np.float64(np.median(polarflux)/normpflux)
    plaw_params, output, Fsyn_model = fit_polarized_flux_plaw(polarnu, polarflux, 
    normpflux, spflux, initialnormal, fitwindow, qmjd, plotout, restnu, oldq,
     oldu, nuorig, normnu)
    #print plaw_params, 'plaw params'
    #print model, 'plawmodels'
    #plaw_params['alpha'].value = plaw_params['alpha'].value -0.1
    print '####################################################################'
#####################Fit Just  a power-law######################################
################################################################################
    logflux = np.log10(fitflux)
    lognu = np.log10(polarnu)
    #fitflux = fitflux-model
    just1plaw_model, just1plaw_output, just1plaw_params=just_single_plaw(fitflux, sfitflux, normalflux, polarnu,  plaw_params, fitwindow)
    just1plaw_chisq = chi_square(just1plaw_model[fitwindow], fitflux[fitwindow],
    sfitflux[fitwindow])
   # print just1plaw_model[fitwindow], fitflux[fitwindow],    sfitflux[fitwindow], 'just1plaw_model[fitwindow], fitflux[fitwinow], sfitflux[fitwindow]'
    just1plaw_reduced_chi = just1plaw_chisq/(np.float(len(fitwindow[0])) - 2)
    print 'just1plaw_reduced_chi', just1plaw_reduced_chi
    singleplawchisquare = np.append(singleplawchisquare, just1plaw_reduced_chi)
    lowslope = just1plaw_params['slope'].value - just1plaw_params['slope'].stderr
    lowinter = just1plaw_params['intercept'].value + just1plaw_params['intercept'].stderr
    hislope = just1plaw_params['slope'].value + just1plaw_params['slope'].stderr
    hiinter = just1plaw_params['intercept'].value - just1plaw_params['intercept'].stderr
    lowplaw = np.power(10, lowslope*lognu + lowinter)
    hiplaw = np.power(10, hislope*lognu + hiinter)
    errplaw = np.abs(hiplaw-lowplaw)
    #sfitflux = errplaw
    #Fit a powerlaw to the flux and use the uncertainties to determine error bar in individual chanels#
    logflux = np.log10(fitflux)
    lognu = np.log10(polarnu)
    slope, intercept, rval, pval, errormat = stats.linregress(lognu, logflux)
    print '################# linear regression################', slope, intercept, rval, pval, errormat
########### Fit the polarized flux with a modified power-law###################
##########  F = (A * nu + B) * Norm * nu^(-alpha + beta*nu) ##############################
########## Start with fitting the polarization with P = (A nu + B)##############
    linepol_params, linepol_output, linepol_model, modified_plaw_params,modified_plaw_model =fit_polarized_flux_modified_plaw_lin(restnu, polarnu, 
    Fsyn, spflux, fitwindow, normflux, plotout, qmjd, oldq, oldu, 
    nuorig, p, sp,plaw_params)

########### Fit the polarized flux with a modified power-law####################
########### of the form F = (A * nu^2 + B*nu + C) * NORM * nu^(-alpha)##########
    #quadpol_params, quadpol_output, quadpol_model, modified_plaw_params_quad, modified_plaw_model_quad = fit_polarized_flux_modified_plaw_quad(restnu, polarnu, 
   # polarflux, spflux, fitwindow, normflux, plotout, qmjd, oldq, oldu, 
   # nuorig, p, sp, plaw_params)
    #print quadpol_params, 'quadpol_params'
################################################################################


###############################################################################
# Perform an F-test on the two models for the polarized flux###################
    chisquareplaw = chi_square(Fsyn_model, polarflux, spflux)
    chisquarelinear = chi_square(modified_plaw_model, polarflux, spflux)
    chiratio1 = chisquareplaw/chisquarelinear
    dof1 = len(Fsyn_model) - 2
    dof2 = len(Fsyn_model) - 3
    dof3 = len(Fsyn_model) - 4
    #red_chisquareplaw = chisquareplaw/dof1
    #red_chisquarelinear = chisquarelinear/dof2
    #sf_polarized_flux = stats.f.sf(chiratio1, dof1, dof2)
    #polarized_flux_sf = np.append(polarized_flux_sf, sf_polarized_flux)
    #plawchi = np.append(plawchi, red_chisquareplaw)
    #linearchi = np.append(linearchi, red_chisquarelinear)
################################################################################
    twocomp_params = Parameters()  
####################Perform the fit for the two component model#################
    modelflux, modelsyn, synmodel, mymodel, bumpmodel, model_chisq,  model_reduced_chi, twocomp_params, model_for_spectrum, mydatamodel, df1, modelbump, sigsyn, sigbump, sigtotal =fit_flux_two_comp_model_mc(restnu, polarnu, Fsyn_model, spflux, fitwindow, normflux, plotout, qmjd, oldq, oldu, nuorig, fitflux, plaw_params, normalflux, normnu, sfitflux)
    mybbbindex = np.append(mybbbindex, twocomp_params['bumpindex'].value) 
    #print 'modelbump', modelbump
    #print (synmodel-nuspec)/nuspec
    bbb_integratedflux= np.append(bbb_integratedflux, modelbump)
    synchrotron_integratedflux=np.append(synchrotron_integratedflux, modelsyn)
    total_integratedflux = np.append(total_integratedflux, modelflux)
    sigsynarr = np.append(sigsynarr, sigsyn)
    sigbumparr = np.append(sigbumparr, sigbump)
    sigfluxarr = np.append(sigfluxarr, sigtotal)
    chisquare_two_comp_model = np.append(chisquare_two_comp_model,  model_chisq/(len(fitwindow[0])-3)) 
    print  model_chisq/(len(fitwindow[0])-3), 'twocomp model chisquare'
    
    #Project the synchrotron into the IR
    logsmodel = np.log10(nuspec)
    #logsmodel = np.log10(synmodel)
    lognu = np.log10(restnu)

    slope, intercept, rval, pval, errormat = stats.linregress(lognu, logsmodel)
    
    #aa = np.polyfit(lognu, logsmodel, 1)
    #slope = aa[0]
    #intercept = aa[1]
    #print slope, intercept
    #plt.plot( lognu, logsmodel)
    #plt.show()
    jflux = 10.0**(slope * np.log10(rest_jnu) + intercept)
    hflux = 10.0**(slope * np.log10(rest_hnu ) + intercept)
    kflux = 10.0**(slope * np.log10(rest_knu ) + intercept)
    #integrate to get the projected measured synflux
    jphot = np.trapz(jflux * J_resp['Jres']*rest_jnu, rest_jnu)/np.trapz( J_resp['Jres']*rest_jnu, rest_jnu)
    hphot = np.trapz(hflux * H_resp['Hres']*rest_hnu, rest_hnu)/np.trapz( H_resp['Hres']*rest_hnu, rest_hnu)
    kphot = np.trapz(kflux * K_resp['Kres']*rest_knu, rest_knu)/np.trapz( K_resp['Kres']*rest_knu, rest_knu)
    print np.trapz(jflux, rest_jnu), 'jphot'
    print jphot/1.0e-23
    print hphot/1.0e-23
    print kphot/1.0e-23
    J_proj = np.append(J_proj, jphot/1.0e-23)
    H_proj = np.append(H_proj, hphot/1.0e-23)
    K_proj = np.append(K_proj, kphot/1.0e-23)
################Perform the two-component fit with the alpha obtained########
################# By fitting the polarized flux with the modified Plaw#######
 #   twocomp_params_2, model_for_spectrum_2, mymodel_2, mydatamodel_2, synmodel_2,bumpmodel_2, modelflux_2, modelsyn_2, modelbump_2, sig_modelflux_2, sig_modelsyn_2, sig_modelbump_2 = two_comp_modified_plaw_mc(fitflux, sfitflux, normalflux, polarnu, normnu, plaw_params, twocomp_params, restnu, modified_plaw_params, fitwindow, qmjd, normflux)   
 #   bbb_integratedflux2= np.append(bbb_integratedflux2, modelbump_2)
 #   synchrotron_integratedflux2=np.append(synchrotron_integratedflux2, 
 #   modelsyn_2)
 #   total_integratedflux2 = np.append(total_integratedflux2, modelflux_2)
 #   sigbbb_integratedflux2= np.append(sigbbb_integratedflux2, sig_modelbump_2)
 #   sigsynchrotron_integratedflux2=np.append(sigsynchrotron_integratedflux2, 
 #   sig_modelsyn_2)
 #   sigtotal_integratedflux2 = np.append(sigtotal_integratedflux2, sig_modelflux_2)

 ################Perform the two-component fit with the alpha obtained########
 ################# By fitting the polarized flux #######
 ################# and holding the big blue bump fixed##########################
    #find the normalization to fix the bbb to
    synmodel_fixedbbb, bumpmodel_fixedbbb, modelsyn_fixedbbb, modelssyn_fixedbbb, modelbump_fixedbbb, modelflux_fixedbbb, mydatamodel_fixedbbb= two_comp_modified_plaw_fixedbbb(fitflux, sfitflux, normalflux, polarnu, normnu, plaw_params, twocomp_params, restnu, modified_plaw_params, fitwindow)
   # print 'modelbump', modelbump_fixedbbb
    bbb_integratedflux_fixedbbb= np.append(bbb_integratedflux_fixedbbb, modelbump_fixedbbb)
    synchrotron_integratedflux_fixedbbb=np.append(synchrotron_integratedflux_fixedbbb, modelsyn_fixedbbb)
    total_integratedflux_fixedbbb = np.append(total_integratedflux_fixedbbb,     modelflux_fixedbbb)   

    


################################################################################
 ###################Perform the Two-Component fit with a BB on top##############
    #perform the fit for the two component model.
    #first select the windows for the fit
    name='blackbody'
   #modelflux, modelsyn, synmodel, mymodel, bumpmodel, model_chisq,  model_reduced_chi, twocomp_params, model_for_spectrum, mydatamodel, df1, modelbump, sigsyn, sigbump, sigtotal
    blackbodyfluxmodel,  twocomp_BB_params, synmodelwbb, bumpmodelwbb, blackbodymodel, synfluxmodel, bbbumpfluxmodel, mychisquarewbb, mmodel, sigbbbumpfluxmodel, sigblackbodyfluxmodel, sigsynfluxmodel, sigfluxmodel= fit_flux_two_comp_wbb_mc(restnu, polarnu, Fsyn_model, spflux, fitwindow, normflux, plotout, qmjd, oldq, oldu, nuorig, fitflux, plaw_params, normalflux, normnu, sfitflux, twocomp_params, nuspec,name, just1plaw_model)
    bb_normalarray= np.append(bb_normalarray, twocomp_BB_params['blackbody_norm'].value)
    totalfluxwbb = np.append(totalfluxwbb, synfluxmodel+bbbumpfluxmodel+blackbodyfluxmodel)
    bbbnormalize = np.append(bbbnormalize, twocomp_BB_params['blackbody_norm'].value)
    plot_spectra_orig(origcorrected, thisspec, nuorig, nuspec, restwave, restnu, plotout, qmjd, normallambdaflux, normalflux,  name='two_comp_model_minusbbb_spectra.png')
    #project the synchrotron component into the IR
    logsmodel = np.log10(synmodelwbb)
    lognu = np.log10(restnu)
    slope, intercept, rval, pval, errormat = stats.linregress(lognu, logsmodel)
    jflux = 10.0**(slope * np.log10(rest_jnu) + intercept)
    hflux = 10.0**(slope * np.log10(rest_hnu ) + intercept)
    kflux = 10.0**(slope * np.log10(rest_knu ) + intercept)
    #integrate to get the projected measured synflux
    jphot = np.trapz(jflux * J_resp['Jres']*rest_jnu, rest_jnu)/np.trapz( J_resp['Jres']*rest_jnu, rest_jnu)
    hphot = np.trapz(hflux * H_resp['Hres']*rest_hnu, rest_hnu)/np.trapz( H_resp['Hres']*rest_hnu, rest_hnu)
    kphot = np.trapz(kflux * K_resp['Kres']*rest_knu, rest_knu)/np.trapz( K_resp['Kres']*rest_knu, rest_knu)
   # print np.trapz(jflux, rest_jnu), 'jphot'
   # print jphot/1.0e-23
   # print hphot/1.0e-23
   # print kphot/1.0e-23
    J_proj_wbb = np.append(J_proj_wbb, jphot/1.0e-23)
    H_proj_wbb = np.append(H_proj_wbb, hphot/1.0e-23)
    K_proj_wbb = np.append(K_proj_wbb, kphot/1.0e-23)

    #Fit a power-law to the flux minus BBB
    myfluxnow, smyfluxnow = bin_array( nuspec-bumpmodelwbb, numberofbins)
    just1plaw_model_minusbbb, just1plaw_output_minusbbb, just1plaw_params_minusbbb=just_single_plaw(myfluxnow, smyfluxnow, normalflux, polarnu,  plaw_params, fitwindow)
    minusbbbalpha = np.append(minusbbbalpha, just1plaw_params_minusbbb['slope'].value*(-1))
    sigmaminusbbbalpha = np.append(sigmaminusbbbalpha, just1plaw_params_minusbbb['slope'].stderr)
    plot_spectra_orig(origcorrected, thisspec-(bumpmodel_fixedbbb*c/restwave**2), nuorig, nuspec-bumpmodel_fixedbbb, restwave, restnu
    , plotout, qmjd, normallambdaflux, normalflux,  name='two_comp_model_minusfixedbbb_spectra.png')
    # determine the size of the hotspot with the flux and temperature.  
    nubbarray = np.linspace(10,20,num=1000)
    nubbarray = 10.0**nubbarray
    totalbbspec = bb(nubbarray, twocomp_BB_params['temperature'].value)
    totalbbflux = np.trapz(totalbbspec, nubbarray)
    observedbbspec = bb(polarnu,  twocomp_BB_params['temperature'].value)
    observedbbflux = np.trapz(observedbbspec, polarnu)*(-1)
    fractionobservedbb = observedbbflux/totalbbflux
    observebbfrac = np.append(observebbfrac, fractionobservedbb)
    print 'total bb flux', totalbbflux, observedbbflux
 
   
   
    #Fit a power-law to the flux minus fixed BBB
    myfluxnow, smyfluxnow = bin_array(nuspec-bumpmodel_fixedbbb, numberofbins)
    binfixbbb, sbinfixbbb = bin_array(bumpmodel_fixedbbb, numberofbins)
    just1plaw_model_minusfixed, just1plaw_output_minusfixed, just1plaw_params_minusfixed=just_single_plaw(myfluxnow,  smyfluxnow, normalflux, polarnu,  plaw_params, fitwindow)
    total_model_fixed_bbb = just1plaw_model_minusfixed + binfixbbb
   # plt.plot(restnu, nuspec)
   # plt.plot(restnu, nuspec-bumpmodel_fixedbbb)
   # plt.plot(polarnu, just1plaw_model_minusfixed)
   # plt.plot(polarnu, total_model_fixed_bbb)
   # plt.plot(polarnu, binfixbbb)
   # plt.show()
    minusfixedbbb_chisq = chi_square(total_model_fixed_bbb[fitwindow], fitflux[fitwindow],
    sfitflux[fitwindow])
    print 'arrays of model' , total_model_fixed_bbb[fitwindow], fitflux[fitwindow],    sfitflux[fitwindow]
    print 'minusfixedbbb_chisq', minusfixedbbb_chisq/(len(fitwindow[0])-2)
    chisquare_minus_fixedbbb = np.append(chisquare_minus_fixedbbb,  minusfixedbbb_chisq/(len(fitwindow[0])-2)) 
    minusfixedbbbalpha = np.append(minusfixedbbbalpha, just1plaw_params_minusfixed['slope'].value*(-1))
    print '#############################################################################'
    print 'minusfixedbbbalpha', just1plaw_params_minusfixed['slope'].value
    print '##############################################################################'
    sigmaminusfixedbbbalpha = np.append(sigmaminusfixedbbbalpha, just1plaw_params_minusfixed['slope'].stderr)
   # print twocomp_BB_params['bumpnormal'], 'two component bb params'
  #### If there is lambda dependent polarization determine what the functional form ################33
############## would be#################################################################
    myratiospecnow = polarflux/myfluxnow
    myratiospecplot = polspec/bumpmodel_fixedbbb
    smyratiospecnow = myratiospecnow*0.01
    just1plaw_model_lambdadep, just1plaw_output_lambdadep, just1plaw_params_lambdadep=just_single_plaw(myratiospecnow,  smyratiospecnow, normalflux, polarnu,  plaw_params, fitwindow)
    just1plaw_chisq = chi_square(just1plaw_model[fitwindow], fitflux[fitwindow],
    sfitflux[fitwindow])
    lambdadepalpha = np.append(lambdadepalpha, just1plaw_params_lambdadep['slope'].value*(-1))
    siglambdadepalpha = np.append(siglambdadepalpha, just1plaw_params_lambdadep['slope'].stderr)
    newwave, snewwave = bin_array(restwave, numberofbins)
   # print len(myratiospecnow), len(newwave),'len(myratiospecnow)'
  #print myratiospecnow, newwave
    plt.plot(polarnu/gom(polarnu), myratiospecnow)
    plt.plot(polarnu/gom(polarnu), just1plaw_model_lambdadep)
    plt.xlabel(r'$\nu \; \times \mathrm{10^15 (Hz)}')
    plt.ylabel(r'Flux Ratio') 
    plt.savefig(plotout+'Ratio_of_spectra.png')
    plt.cla()
    plt.close()
    #print 100*polspec/origcorrected
    fig = plt.figure(999, figsize=(10,7)) 
    plt.plot(restnu/gom(polarnu), 100*polspec/nuorig)
    plt.plot(polarnu/gom(polarnu),100*p,'o', ls='', color=(0,1, 1))
    plt.title('MJD = '+str(qmjd))
    plt.xlabel(r'$\nu \; ( \mathrm{10^{15} Hz)}$')
    plt.ylabel(r'$P\; (\%)$ ')
    plt.savefig(plotout+'Pol_percent.png')
    plt.cla()
    plt.close()
    
    plt.plot(polarnu/gom(polarnu), polarflux/sfitflux, color=(0,0,1))
    plt.title('MJD = '+str(qmjd))
    plt.ylabel(r'$F_P/\sigma F\; $ ')
    plt.xlabel(r'$\nu \; ( \mathrm{10^{15} Hz)}$')
    plt.savefig(plotout+'Pol_over_sigma.png')
    plt.cla()
    plt.close()
    bbsynmodel = np.append(bbsynmodel, synfluxmodel)
    bbbumpmodel = np.append(bbbumpmodel, bbbumpfluxmodel)  
    blackbodyflux = np.append(blackbodyflux, blackbodyfluxmodel)
    sigbbbumpfluxmodel_wbb =np.append(sigbbbumpfluxmodel_wbb, sigbbbumpfluxmodel)
    sigblackbodyfluxmodel_wbb = np.append(sigblackbodyfluxmodel_wbb, sigblackbodyfluxmodel)
    sigsynfluxmodel_wbb =  np.append(sigsynfluxmodel_wbb, sigsynfluxmodel)
    modelwblackbody = blackbodymodel+synmodelwbb+bumpmodelwbb
    sigfluxmodel = NtoF*(synfluxmodel+bbbumpfluxmodel+blackbodyfluxmodel)
    sigfluxarr_wbb = np.append(sigfluxarr_wbb, sigfluxmodel)
    #$mychisquarewbb =chi_square(modelwblackbody[fitwindow], fitflux[fitwindow], sfitflux[fitwindow])
    #print 'modelwblackbody, fitflux, sfitflux'
    #print modelwblackbody[fitwindow], fitflux[fitwindow], sfitflux[fitwindow]
    dofhere = len(fitwindow[0]) - 5
   # print (mmodel[fitwindow]-fitflux[fitwindow]), sfitflux[fitwindow], mychisquarewbb
    #print modelwblackbody[fitwindow], fitflux[fitwindow], sfitflux[fitwindow]
    #print np.sum((modelwblackbody[fitwindow]-fitflux[fitwindow])**2/sfitflux[fitwindow]**2)
    temperature_array = np.append(temperature_array, twocomp_BB_params['temperature'].value)
    temperature_error = np.append(temperature_error, twocomp_BB_params['temperature'].stderr)
    blackbodychisquare = np.append(blackbodychisquare, mychisquarewbb)

 ################Perform the two-component fit with the alpha obtained########
 ################# By fitting the polarized flux with the modified Plaw#######
 ################# and adding a blackbody ##########################
 #   name='blackbody_modifiedplaw'
 #   modified_plaw_params['alpha'].value = just1plaw_params['slope'].value- plaw_params['alpha'].value
 #   blackbodyfluxmodel_modplaw,  twocomp_BB_params_modplaw, synmodelwbb_modplaw, bumpmodelwbb_modplaw, blackbodymodel_modplaw, bbsynfluxmodel_modplaw, bbbumpfluxmodel_modplaw, mychisquarewbb_modplaw, mmodelmod,  sigbbbumpfluxmodel_mod, sigblackbodyfluxmodel_mod, sigsynfluxmodel_mod, sigfluxmodel_mod= fit_flux_two_comp_wbb_mc(restnu, polarnu, polarflux, spflux, fitwindow, normflux, plotout, qmjd, oldq, oldu, nuorig, fitflux, modified_plaw_params, normalflux, normnu, sfitflux, twocomp_params_2, nuspec,name, just1plaw_model)
  #  bbsynmodel_modplaw = np.append(bbsynmodel_modplaw, bbsynfluxmodel_modplaw)
  #  plot_spectra_orig(origcorrected, thisspec, nuorig, nuspec-bumpmodelwbb_modplaw, restwave, restnu
  #  , plotout, qmjd, normallambdaflux, normalflux, name='two_comp_model_minusbbb_modplaw_spectra.png' )
  #  bbbumpmodel_modplaw = np.append(bbbumpmodel_modplaw, bbbumpfluxmodel_modplaw) 
  #  totalmodelherenow = blackbodymodel_modplaw+synmodelwbb_modplaw+bumpmodelwbb_modplaw
  #  totalfluxwbb_modplaw = np.append(totalfluxwbb_modplaw, bbsynfluxmodel_modplaw+bbbumpfluxmodel_modplaw+blackbodyfluxmodel_modplaw)     
  #  blackbodyflux_modplaw = np.append(blackbodyflux_modplaw, blackbodyfluxmodel_modplaw)
  #  sigbbsynmodel_modplaw = np.append(sigbbsynmodel_modplaw, sigsynfluxmodel_mod)
  #  sigbbbumpmodel_modplaw = np.append(sigbbbumpmodel_modplaw,  sigbbbumpfluxmodel_mod) 
  #  sigtotalfluxwbb_modplaw = np.append(sigtotalfluxwbb_modplaw, sigfluxmodel_mod)     
  #  sigblackbodyflux_modplaw = np.append(sigblackbodyflux_modplaw, sigblackbodyfluxmodel_mod)
  #  temperature_array_modplaw = np.append(temperature_array_modplaw, twocomp_BB_params_modplaw['temperature'].value)
  #  blackbodychisquare_modplaw = np.append(blackbodychisquare_modplaw, mychisquarewbb_modplaw)
  #  chisquarelinepolwbb = chi_square(mmodelmod[fitwindow], fitflux[fitwindow], sfitflux[fitwindow])
  #  degreesoffreedom = len(fitwindow[0])-4
  #  redchisquare_linearpolwbb = chisquarelinepolwbb/degreesoffreedom
  #  otheralpha=twocomp_BB_params_modplaw['synalpha'].value
    #blackbodychisquare_modplaw = np.append(blackbodychisquare_modplaw, redchisquare_linearpolwbb)
 ###############################################################################


    
##########################
    #############Perform a fit of a power-law to the spectrum##################
    ########### Do an F-Test with this fit and each of the 2 component models###########
    df1 = len(fitwindow[0]) - 3
    df2 = (np.float(len(fitwindow[0])) - 5)
    df3 = len(fitwindow[0]) -2
    chisq1 = chi_square(mydatamodel[fitwindow], fitflux[fitwindow],
    sfitflux[fitwindow])
    redchisq1 = chisq1/df1
    #chisq2 = chi_square(mydatamodel_2[fitwindow], fitflux[fitwindow], 
 #   sfitflux[fitwindow])
  #  print 'chisq2', chisq2, df2
    #redchisq2 = chisq2/df2
    chisq3 = chi_square(just1plaw_model[fitwindow], fitflux[fitwindow],  
    sfitflux[fitwindow])
    redchisq3 = chisq1/df3
    chisq4 = chi_square(mydatamodel_fixedbbb[fitwindow], fitflux[fitwindow],  
    sfitflux[fitwindow])
    onlyplaw_alpha = np.append(onlyplaw_alpha, just1plaw_params['slope'].value*(-1.0))
    sigma_onlyplaw_alpha = np.append(sigma_onlyplaw_alpha, 
    just1plaw_params['slope'].stderr)
################################################################################
############Plot the best fit 2 component models along with the best powerlaw
      #  binfixbbb, sbinfixbbb = bin_array(bumpmodel_fixedbbb, numberofbins)
    binbumpmodel, sbinbumpmodel = bin_array(bumpmodel, numberofbins)
    binsyn, sbinsyn = bin_array(synmodel, numberofbins)
    #print 'synmodel', synmodel
    name = '_bestfitmodels.png'
    plot_two_component_model(restnu, normnu, nuspec, normalflux, polarnu, 
    fitflux, sfitflux, mydatamodel, binsyn, binbumpmodel, 
    just1plaw_model, plotout, qmjd, name, fitwindow, y1)
 
   # name = '_bestfitmodels_linearpol.png'
   # plot_two_component_model(restnu, normnu, nuspec, normalflux, polarnu, 
   # fitflux, sfitflux, mydatamodel_2, synmodel_2, bumpmodel_2, just1plaw_model, 
   # plotout, qmjd, name, fitwindow, y1)
    
    
    name = '_bestfitmodels_fixedbbb.png'
    plot_two_component_model(restnu, normnu, nuspec, normalflux, polarnu, 
    fitflux, sfitflux,  total_model_fixed_bbb, just1plaw_model_minusfixed, 
    binfixbbb, just1plaw_model, plotout, qmjd, name, fitwindow,y1)
    synchrotron_normalize = np.append(synchrotron_normalize, 
    twocomp_params['synnormal'].value*np.power(normnu, 
    twocomp_params['synalpha'].value*(-1))*normflux )
    alpha_array = np.append(alpha_array, plaw_params['alpha'].value)
  #  modalpha_array = np.append(modalpha_array, otheralpha)
    alpha_err = np.append(alpha_err, plaw_params['alpha'].stderr)
    print 'mjd, diff alpha',qmjd,  (just1plaw_params['slope'].value*(-1.0) -  plaw_params['alpha'].value)
    #linearpol_alpha = np.append(linearpol_alpha, twocomp_params_2['synalpha'].value)
   # linearpol_alpha_sigma = np.append(linearpol_alpha_sigma, twocomp_params_2['synalpha'].stderr)
  
#Plot The Spectral Index for various models######################
#Do K-S tests on the spectral index distributions###############
###############################################################################
#name='twocomp'
#print alpha_err, 'alpha error'
#plot_alpha(mjdarr, alpha_array, alpha_err, oldplotout, onlyplaw_alpha, 
#sigma_onlyplaw_alpha, modified_alpha_array, 
#sigma_modified_alpha_array, bbb_integratedflux, name)
#name='linearpol'
#plot_alpha(mjdarr, linearpol_alpha, linearpol_alpha_sigma, oldplotout, onlyplaw_alpha, 
#sigma_onlyplaw_alpha, modified_alpha_array, 
#sigma_modified_alpha_array, bbb_integratedflux, name)
#name='quadpol'
#plot_alpha(mjdarr, quadratic_alpha, quadratic_alpha_sigma, oldplotout, onlyplaw_alpha, 
#sigma_onlyplaw_alpha, modified_alpha_array, 
#sigma_modified_alpha_array, bbb_integratedflux, name)
name='twocomp_w_ebar'
plot_alpha_better(mjdarr, alpha_array, alpha_err, oldplotout, onlyplaw_alpha, sigma_onlyplaw_alpha, name)
twocomp_D, twocomp_p  = ks_2samp(alpha_array, onlyplaw_alpha)
print 'twocomp_p', twocomp_p
#name='modified_plaw_test'
#plot_alpha_better_chi(mjdarr, alpha_array, alpha_err, oldplotout, modalpha_array, sigma_onlyplaw_alpha, name, linearpolwbb_chi)

mod_twocomp_D, mod_twocomp_p = ks_2samp(alpha_array, onlyplaw_alpha)
#name = 'chisquareValpha'
#plot_alphaVchi(mjdarr, alpha_array, modalpha_array, linearpolwbb_chi, name, oldplotout)
name='minusbbb_alphas'
plot_alpha_better(mjdarr, alpha_array, alpha_err, oldplotout, minusbbbalpha, sigmaminusbbbalpha, name)
name='minusfixedbbb_alphas'
plot_alpha_better(mjdarr, alpha_array, alpha_err, oldplotout, minusfixedbbbalpha, sigmaminusfixedbbbalpha, name)
#name = 'minusfixedbbb_lambdadeppol'
#plot_alpha_better(mjdarr, alpha_array, alpha_err, oldplotout, lambdadepalpha, siglambdadepalpha, name)

###############################################################################
###############################################################################
#Plot the survival functions for the various polarized flux models
#plot_polflux_sf(mjdarr, polarized_flux_sf, polarized_flux_sf2, oldplotout, 'polarized_flux_models_ftest')
#plot the reduced chisquare for the various polarized flux 
#plot_chisquare_polarized(mjdarr, plawchi, linearchi, quadchi,oldplotout) #linearchi, quadchi
#plot_chisquare_polarized_better(mjdarr, plawchi, 'Polarized-Flux Power-Law', 'polarized_flux_plaw.png',oldplotout)
#plot_chisquare_polarized_better(mjdarr, linearchi, 'Polarized-Flux Power-Law w Linear-Term', 'polarized_flux_linear.png',oldplotout)
#plot_chisquare_polarized_better(mjdarr, plawchi, 'Polarized-Flux Power-Law w Quadratic-Term', 'polarized_flux_quad.png',oldplotout)
#plot_polflux_sf(mjdarr, ftest_2comp_model, ftest_2comp_model_mplaw, 
#oldplotout, 'polarized_flux_models_ftest')
#plot_chisquare(mjdarr, chi1arr, chi2arr, chi3arr, chi4arr, blackbodychisquare, chi5arr, quadraticpolwbb_chi,  linearpolwbb_chi, oldplotout)
#plot_chisquare_better(mjdarr, exponbb_chi_reduced_arr,'Exponential Cutoff Polarization With Blackbody', 'chisquare_exponsynwbb.png', oldplotout)
#plot_chisquare_better(mjdarr, expon_chi_reduced_arr,'Exponential Polarization', 'chisquare_exponsyn.png', oldplotout)
#plot_chisquare_better(mjdarr, chi1arr,'Two-Component Model', 'chisquare_2compmodel.png', oldplotout)
#plot_chisquare_better(mjdarr, chi2arr,'Linear Polarization', 'chisquare_linearpol.png', oldplotout)
#plot_chisquare_better(mjdarr, chi3arr,'Just Power-Law', 'chisquare_justplaw.png', oldplotout)
#plot_chisquare_better(mjdarr, chi4arr,'Fixed-BBB', 'chisquare_fixedBBB.png', oldplotout)
plot_chisquare_better(mjdarr, chisquare_two_comp_model, 'Two-Component Model', 'chisquare_2compmodel.png', oldplotout)
plot_chisquare_better(mjdarr, blackbodychisquare,'Two-Component Model With Black-Body', 'chisquare_2compmodelwbb.png', oldplotout)
plot_chisquare_better(mjdarr, singleplawchisquare, 'Full Spectrum Just Power-law', 'chisquare_fullspecjustplaw.png', oldplotout)
#plot_chisquare_better(mjdarr,blackbodychisquare_modplaw, 'Modified Power-Law', 'modplaw_chisquare.png', oldplotout)
plot_chisquare_better(mjdarr, chisquare_minus_fixedbbb,'Holding Big Blue Bump Fixed', 'chisquare_fixedbbb.png', oldplotout)
#plot_chisquare_better(mjdarr, linearpolwbb_chi,'Linear Polarization With Black-Body', 'chisquare_linearpolarization.png', oldplotout)

#plot how parameters of interest have changed throughout several years
#plot_the_final_models(mjdarr, bbb_integratedflux2, synchrotron_integratedflux2, 
#total_integratedflux2, oldplotout, bbb_integratedflux_fixedbbb, synchrotron_integratedflux_fixedbbb, total_integratedflux_fixedbbb, total_integratedflux, onlyplaw_alpha, chi1arr, blackbodyflux, temperature_array, bbb_integratedflux, synchrotron_integratedflux, mybbbindex, bbsynmodel, bbbumpmodel, totalfluxwbb, bbb_integratedflux_quad, synchrotron_integratedflux_quad, total_integratedflux_quad,totalfluxwbb_quadpol, bbsynmodel_quadpol, bbbumpmodel_quadpol, blackbodyflux_quadpol,totalfluxwbb_modplaw, bbsynmodel_modplaw, bbbumpmodel_modplaw, blackbodyflux_modplaw, bbb_integratedflux_ec, synchrotron_integratedflux_ec, total_integratedflux_ec,totalfluxwbb_expon, bbsynmodel_expon, bbbumpmodel_expon, blackbodyflux_expon)



name = 'syn_alpha_array'
plot_flux_onlyplaw(total_integratedflux, alpha_array,  mjdarr, oldplotout, name) 
#print 'type of the variables', type(bbbumpmodel), type(sigbbbumpfluxmodel_wbb), type(bbsynmodel), type(sigsynfluxmodel_wbb), type(blackbodyflux), type(sigblackbodyfluxmodel_wbb), type(total_integratedflux), type(sigfluxarr), oldplotout, name
name='two_comp_model_webar'
sigfluxarr = total_integratedflux*NtoF

plot_models_mc(mjdarr, bbb_integratedflux, sigbumparr, synchrotron_integratedflux, sigsynarr, total_integratedflux, sigfluxarr, oldplotout, name)
sigtotal_integratedflux2 = total_integratedflux2*NtoF
#name = 'two_comp_model_modplaw_webar'
#plot_models_mc(mjdarr,  bbb_integratedflux2, sigbbb_integratedflux2, synchrotron_integratedflux2, sigsynchrotron_integratedflux2, #total_integratedflux2,  sigtotal_integratedflux2, oldplotout, name)

#plot_models_mc(mjdarr, bbb_integratedflux, sigbump, synchrotron_integratedflux, sigsyn, total_integratedflux, sigflux, plotout, name):


name ='two_comp_model_wbb_ebar'
plot_modelswbb_mc(mjdarr, bbbumpmodel, sigbbbumpfluxmodel_wbb, bbsynmodel,  sigsynfluxmodel_wbb, blackbodyflux, sigblackbodyfluxmodel_wbb, total_integratedflux, sigfluxarr, oldplotout, name)
spectralcomponentsflux = dirout+'spectral_component_flux.csv'
with open(spectralcomponentsflux, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow('[mjdarr, bbbumpmodel, sigbbbumpfluxmodel_wbb, bbsynmodel,  sigsynfluxmodel_wbb,  blackbodyflux, sigblackbodyfluxmodel_wbb, total_integratedflux, sigfluxarr' )
    for xx in range(len(mjdarr)):
        writer.writerow([mjdarr[xx],bbbumpmodel[xx], sigbbbumpfluxmodel_wbb[xx], bbsynmodel[xx],  sigsynfluxmodel_wbb[xx], blackbodyflux[xx], sigblackbodyfluxmodel_wbb[xx], total_integratedflux[xx], sigfluxarr[xx] ])

synprojoutput= dirout+'synchrotron_projected.csv'
with open(synprojoutput, 'wb') as csvfile:
   spamwriter = csv.writer(csvfile, delimiter=',')
   csvfile.write(' MJD, jproj, hproj, kproj \n')
   for xx in range(len(mjdarr)):
      spamwriter.writerow([mjdarr[xx], J_proj[xx],H_proj[xx], K_proj[xx] ])

synprojoutputwbb= dirout+'synchrotron_wbb_projected.csv'
with open(synprojoutputwbb, 'wb') as csvfile:
   spamwriter = csv.writer(csvfile, delimiter=',')
   csvfile.write(' MJD, jproj, hproj, kproj \n')
   for xx in range(len(mjdarr)):
      spamwriter.writerow([mjdarr[xx], J_proj_wbb[xx],H_proj_wbb[xx], K_proj_wbb[xx] ])



name ='two_comp_model_wbb_ebar_zoom_56500_57000'
zoommjd = np.where((mjdarr > 56500.0) & (mjdarr < 57000.0))
#plot_modelswbb_mc(mjdarr[zoommjd], bbbumpmodel[zoommjd], sigbbbumpfluxmodel_wbb[zoommjd], bbsynmodel[zoommjd],  sigsynfluxmodel_wbb[zoommjd], blackbodyflux[zoommjd], sigblackbodyfluxmodel_wbb[zoommjd], total_integratedflux[zoommjd], sigfluxarr[zoommjd], oldplotout, name)
name ='two_comp_model_wbb_ebar_zoom_55000_55700'
zoommjd = np.where((mjdarr > 55000.0) & (mjdarr < 55700.0))
#plot_modelswbb_mc(mjdarr[zoommjd], bbbumpmodel[zoommjd], sigbbbumpfluxmodel_wbb[zoommjd], bbsynmodel[zoommjd],  sigsynfluxmodel_wbb[zoommjd], blackbodyflux[zoommjd], sigblackbodyfluxmodel_wbb[zoommjd], total_integratedflux[zoommjd], sigfluxarr[zoommjd], oldplotout, name)

name ='two_comp_model_wbb_ebar_zoom_55700_56500'
zoommjd = np.where((mjdarr > 55700.0) & (mjdarr < 56500.0))
#plot_modelswbb_mc(mjdarr[zoommjd], bbbumpmodel[zoommjd], sigbbbumpfluxmodel_wbb[zoommjd], bbsynmodel[zoommjd],  sigsynfluxmodel_wbb[zoommjd], blackbodyflux[zoommjd], sigblackbodyfluxmodel_wbb[zoommjd], total_integratedflux[zoommjd], sigfluxarr[zoommjd], oldplotout, name)

plt.plot(mjdarr, J_proj_wbb, 'o', color = (1,0,0))
plt.plot(mjdarr, H_proj_wbb, 'o', color=(0,0,0))
plt.plot(mjdarr, K_proj_wbb, 'o', color=(0,1,0))
plt.show()
#plot the synchrotron model vs spectral index of the total spectrum
plt.plot(bbsynmodel, onlyplaw_alpha, 'o', color=(0,0,1))
plt.xlabel(r'Synchrotron Flux')
plt.ylabel(r'$\alpha_{spectrum}$')
plt.savefig('synchrotron_vs_alpha.png')
plt.cla()
plt.close()

#print bbsynmodel/total_polarized_flux
#plt.plot(mjdarr, bbsynmodel/total_polarized_flux)
#plt.show()
#name ='two_comp_modelmod_wbb_ebar'
#sigtotalfluxwbb_modplaw = totalfluxwbb_modplaw * NtoF
#plot_modelswbb_mc(mjdarr, bbbumpmodel_modplaw, sigbbbumpmodel_modplaw, bbsynmodel_modplaw,  sigbbsynmodel_modplaw,  blackbodyflux_modplaw, sigblackbodyflux_modplaw, totalfluxwbb_modplaw,sigtotalfluxwbb_modplaw, oldplotout, name)

 #myfluxnow, smyfluxnow = bin_array(nuspec-bumpmodel_fixedbbb, numberofbins)
  #  binfixbbb, sbinfixbbb = bin_array(bumpmodel_fixedbbb, numberofbins)
   # just1plaw_model_minusfixed, just1plaw_output_minusfixed, just1plaw_params_minusfixed=just_single_plaw(myfluxnow,  smyfluxnow, normalflux, polarnu,  plaw_params, fitwindow)
   # total_model_fixed_bbb = just1plaw_model_minusfixed + binfixbbb
name = 'model_w_bbb_fixed'
plot_modelswbb_mc(mjdarr, bbb_integratedflux_fixedbbb, bbb_integratedflux_fixedbbb*0.1, synchrotron_integratedflux_fixedbbb,  synchrotron_integratedflux_fixedbbb*0.1,  0.0*synchrotron_integratedflux_fixedbbb, synchrotron_integratedflux_fixedbbb*0.0, total_integratedflux_fixedbbb ,total_integratedflux_fixedbbb*0.1, oldplotout, name)
#name='two_comp_modelmod_fixedbbb_ebar'
#plot_modelswbb_mc(mjdarr,bbb_integratedflux_fixedbbb, )


#(mjdarr, bbb_integratedflux2, synchrotron_integratedflux2, 
#total_integratedflux2, oldplotout, bbb_integratedflux_fixedbbb, synchrotron_integratedflux_fixedbbb, total_integratedflux_fixedbbb, total_integratedflux, onlyplaw_alpha, chi1arr, blackbodyflux, temperature_array, bbb_integratedflux, synchrotron_integratedflux, mybbbindex, bbsynmodel, bbbumpmodel, totalfluxwbb, bbb_integratedflux_quad, synchrotron_integratedflux_quad, total_integratedflux_quad,totalfluxwbb_quadpol, bbsynmodel_quadpol, bbbumpmodel_quadpol, blackbodyflux_quadpol,totalfluxwbb_modplaw, bbsynmodel_modplaw, bbbumpmodel_modplaw, blackbodyflux_modplaw, bbb_integratedflux_ec, synchrotron_integratedflux_ec, total_integratedflux_ec,totalfluxwbb_expon, bbsynmodel_expon, bbbumpmodel_expon, blackbodyflux_expon)

coeff = pearsonr(polarization_slopes, bbb_integratedflux/total_integratedflux)
print coeff
plt.plot(polarization_slopes, bbb_integratedflux/total_integratedflux, 'o')
plt.xlabel('Slope of Polarized Flux')
plt.ylabel('Ratio of BBB Flux to Total Flux')
plt.savefig(oldplotout+'BBBvpolslope.png')
plt.cla()
plt.close()

plt.plot(mjdarr, temperature_array, 'o', color= (0,0,0))
plt.errorbar(mjdarr,  temperature_array, yerr=temperature_error, marker='o', color = (0,0,0), ls= '')
plt.ylim(0,80000)
plt.xlabel('MJD')
plt.ylabel(r'$T_{BB} $(K)')
plt.savefig(oldplotout+'BBB_TEMP.png')
plt.cla()
plt.close()
#figure out the size of the blackbody
steffanboltzman = 5.67e-5
sourcedistance = 2423.1*3.08568e24
#sourcedistance = 6009.9*3.08568e24
#sourcedistance= 760.5*3.08568e24
sizeofbbody = sourcedistance*np.sqrt(2.0*blackbodyflux/steffanboltzman)*(1.0/temperature_array)**2
sizeofbbody = sizeofbbody/1.496e+13
bbsizeerror= (2 * temperature_error/temperature_array) * sizeofbbody
print sizeofbbody
plt.plot(mjdarr, sizeofbbody, 's', color=(0,0,0))
plt.errorbar(mjdarr, sizeofbbody, yerr = bbsizeerror, marker= 's', color=(0,0,0), ls='')
plt.ylim(0,400)
plt.xlabel(r'MJD')
plt.ylabel(r'Size of Black Body (AU)')
plt.savefig(oldplotout+'blackbodysize.png')
#plt.show()
#plot histograms of the total synchrotron flux and the total BBBflux
#plt.hist(bbb_integratedflux/gom(bbb_integratedflux), 10, facecolor='b')
#plt.xlabel(r'Integrated BBB Flux ($10^{'+str(np.log10(gom(bbb_integratedflux)))+'}$ erg s$^{-1}$ cm$^{-2}$)')
#plt.ylabel(r'Frequency')
#plt.savefig(oldplotout+'BBB_flux_hist.png')
#plt.cla()
#plt.close()
#plt.hist(synchrotron_integratedflux/gom(bbb_integratedflux), 10, facecolor='r')
#plt.xlabel(r'Integrated Synchrotron Flux ($10^{'+str(np.log10(gom(bbb_integratedflux)))+'}$ erg s$^{-1}$ cm$^{-2}$)')
#plt.ylabel(r'Frequency')
#plt.savefig(oldplotout+'syn_flux_hist.png')
#plt.cla()
#plt.close()
#plot histograms of the total synchrotron flux and the total blackbody flux for that version of the model
#plot_modelswbb_mc(mjdarr, bbbumpmodel, sigbbbumpfluxmodel_wbb, bbsynmodel,  sigsynfluxmodel_wbb, blackbodyflux, sigblackbodyfluxmodel_wbb, total_integratedflux, sigfluxarr, oldplotout, name)
#plt.hist((bbbumpmodel+blackbodyflux)/gom(bbb_integratedflux), 10, facecolor='b')
#plt.xlabel(r'Integrated BBB Flux ($10^{'+str(np.log10(gom(bbb_integratedflux)))+'}$ erg s$^{-1}$ cm$^{-2}$)')
#plt.ylabel(r'Frequency')
#plt.savefig(oldplotout+'BBB_wbb_flux_hist.png')
#plt.cla()
#plt.close()
#plt.hist(bbsynmodel/gom(bbb_integratedflux), 10, facecolor='r')
#plt.xlabel(r'Integrated Synchrotron Flux ($10^{'+str(np.log10(gom(bbb_integratedflux)))+'}$ erg s$^{-1}$ cm$^{-2}$)')
#plt.ylabel(r'Frequency')
#plt.savefig(oldplotout+'syn_flux_wbb_hist.png')
#plt.cla()
#plt.close()


#make a histogram of the change in polarization
poldiff_arr = poldiff_arr * 100.
plt.hist(poldiff_arr, 15, facecolor='b')
plt.xlabel(r'$\Delta\/ \Pi \mathrm{(\%)}$')
plt.ylabel(r'N')
plt.savefig(oldplotout+'lam_dep_pol_hist.png')
plt.cla()
plt.close()
#print 'blackbody normal const', bb_normalarray

print alpha_array, 'alpha array'
print onlyplaw_alpha, 'only plaw alpha'
print 'median difference', np.median(alpha_array-onlyplaw_alpha)
print ' MJD, alpha, sigma'
for xx in range(len(mjdarr)):
      print mjdarr[xx], alpha_array[xx], alpha_err[xx], onlyplaw_alpha[xx], sigma_onlyplaw_alpha[xx]

alphaoutput = dirout+'alphaoutput.csv'
with open(alphaoutput, 'wb') as csvfile:
   spamwriter = csv.writer(csvfile, delimiter=',')
   csvfile.write(' MJD, alpha, sigma \n')
   for xx in range(len(mjdarr)):
      spamwriter.writerow([mjdarr[xx], alpha_array[xx], alpha_err[xx], onlyplaw_alpha[xx], sigma_onlyplaw_alpha[xx] ])



alphaoutputminusfixed = dirout+'alphaoutputminusfixedbbb.csv'
with open(alphaoutputminusfixed, 'wb') as csvfile:
   spamwriter = csv.writer(csvfile, delimiter=',')
   csvfile.write(' MJD, alpha, sigma \n')
   for xx in range(len(mjdarr)):
      spamwriter.writerow([mjdarr[xx], alpha_array[xx], alpha_err[xx],  minusfixedbbbalpha[xx], sigmaminusfixedbbbalpha[xx]])

alphaoutputminusfixed = dirout+'alphaoutputlambdadeppol.csv'
with open(alphaoutputminusfixed, 'wb') as csvfile:
   spamwriter = csv.writer(csvfile, delimiter=',')
   csvfile.write(' MJD, alpha, sigma \n')
   for xx in range(len(mjdarr)):
      spamwriter.writerow([mjdarr[xx], alpha_array[xx], alpha_err[xx], lambdadepalpha[xx], siglambdadepalpha[xx]])

bbboutput = dirout+'bbboutput.csv'
print 'bbbnormalize'
print bbbnormalize, np.mean(bbbnormalize)
with open(bbboutput, 'wb') as csvfile:
   spamwriter = csv.writer(csvfile, delimiter=',')
   csvfile.write(' MJD, bbbnormalize \n')
   for xx in range(len(mjdarr)):
      spamwriter.writerow([mjdarr[xx], bbbnormalize[xx]])


ksresult = dirout+'KS_test_alpha.csv'
with open(ksresult, 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    csvfile.write(' twocomp_D, twocomp_p, modtwocomp_d, modtwocomp_p \n')
    spamwriter.writerow([twocomp_D, twocomp_p, mod_twocomp_D, mod_twocomp_p])

