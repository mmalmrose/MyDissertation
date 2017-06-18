#!/usr/bin/python
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
from scipy.stats import ks_2samp
from scipy.stats import pearsonr
import matplotlib.gridspec as gridspec
from blackbody import blackbody as bb
#=====================================================================================

#optical_comp_flux_file = '/media/mmalmrose/Lightning/Smith_data/outputfiles/1222+216/VARBBB/textout/spectral_component_flux.csv'
#gamma_ray_flux_file = '/media/mmalmrose/Lightning/Smith_data/outputfiles/1222+216/VARBBB/textout/PKSB1222+216_604800.fits'
#optical_comp_flux_file = '/media/mmalmrose/Lightning/Smith_data/outputfiles/OJ248/VARBBB/textout/spectral_component_flux.csv'
#gamma_ray_flux_file = '/media/mmalmrose/Lightning/Smith_data/outputfiles/OJ248/VARBBB/textout/0827+243_604800.lc'
#optical_comp_flux_file = '/media/mmalmrose/Lightning/Smith_data/outputfiles/3C273/VARBBB/textout/spectral_component_flux.csv'
#gamma_ray_flux_file = '/media/mmalmrose/Lightning/Smith_data/outputfiles/3C273/VARBBB/textout/3C273_604800.lc'
optical_comp_flux_file = '/media/mmalmrose/Lightning/Smith_data/outputfiles/CTA102/VARBBB/textout/spectral_component_flux.csv'
gamma_ray_flux_file = '/media/mmalmrose/Lightning/Smith_data/outputfiles/CTA102/VARBBB/textout/CTA102_604800.lc'
#read in the gamma rays

#gamma_ray_dat = np.genfromtxt(gamma_ray_flux_file, skip_header=3, delimiter=',', dtype=[('mjd', 'f'), ('flux', 'f'), ('sflux', 'f'), ('binwindow', 'f'), ('s_o_binwindow', 'f'), ('expt', 'f')] )
optical_components = np.genfromtxt(optical_comp_flux_file, skip_header=1, delimiter=',', dtype=[('mjd', 'f'), ('bbbmod', 'f'), ('sbbb' , 'f'), ('synchrotron', 'f'), ('ssynchrotron', 'f'), ('blackbody', 'f'), ('sblackbody', 'f'), ('total_flux', 'f'), ('stotal_flux', 'f') ])

#read in fermi weekly lightcurve
lc = pyfits.open(gamma_ray_flux_file)
gamma_ray_dat = lc[1].data
#print gamma_ray_dat
#convert the date from seconds since Jan 1, 2001 to MJD
timecol = gamma_ray_dat['START'] + (0.5)*(gamma_ray_dat['STOP'] - gamma_ray_dat['START'])
timecol = (timecol/86400)+2451910.500000-2400000.0
print timecol

#plot the lightcurves
fig = plt.figure(6, figsize=(10,7))
gs = gridspec.GridSpec(2,1, height_ratios=[1,1])
ax1 = plt.subplot(gs[0])
ax1.errorbar(timecol, gamma_ray_dat['FLUX_100_300000']/1.0e-7, yerr=gamma_ray_dat['ERROR_100_300000']/1.0e-7, linestyle= '', fmt='o')
ax1.set_ylabel(r'$F_{\gamma} \mathrm{(\times 10^{-7} \ phot\ s^{-1} \ cm^{-2})}$' )
ax2 = plt.subplot(gs[1])
ax2.errorbar(optical_components['mjd'], optical_components['blackbody']/1.0e-13, yerr=optical_components['sblackbody']/1.0e-13, linestyle='', fmt='s')
ax2.set_xlabel('MJD')
ax2.set_ylabel(r'$F_{BB} \mathrm{(\times 10^{-13}erg\/s^{-1}\/cm^{-2})}$')
ax2.set_xlim(54500,58000)
plt.savefig('1222+216_gammaray_hotspot_flux.eps', format='eps')
plt.close()
plt.cla()
#Apply the DCF of Edelson and Krolik (1988)

gamarray = gamma_ray_dat['FLUX_100_300000']
gamerror = gamma_ray_dat['ERROR_100_300000']
gammean = np.mean(gamarray)
gamstdev = np.std(gamarray)
gamdate = timecol
bumparray = optical_components['blackbody']
bumperror = optical_components['sblackbody']
bumpmean = np.mean(bumparray)
bumpstdev = np.std(bumparray)
bumpdate = optical_components['mjd']
#create an array to create the udcf and time offsets for each pair
UDCF = np.array([])
deltat = np.array([])
gamsize = len(gamarray)
bumpsize = len(bumparray)
#print gamerror/gamstdev
#print bumperror, bumpstdev
i = 0
for i in range(gamsize-1):
    for thisi in range(bumpsize - 1):
        top = (bumparray[thisi] - bumpmean) * (gamarray[i] - gammean)
        bot =(bumpstdev**2 - bumperror[thisi]**2)*(gamstdev**2 - gamerror[i]**2)
        bot = np.sqrt(bot)
        timediff = gamdate[i]-bumpdate[thisi]
        UDCF = np.append(UDCF, top/bot)
        deltat = np.append(deltat, timediff)


#print UDCF
#print np.max(deltat), np.min(deltat)
plt.plot(deltat, UDCF, 'o')
plt.xlabel(r'$\Delta$ T (days)')
plt.ylabel(r'UDCF')
plt.savefig('UDCF_plot.eps', format='eps')
#print deltat
#plt.show()
plt.close()
plt.cla()
#Take the Unbinned DCF and bin it
bindates = np.arange( np.min(deltat),  np.max(deltat), 30)
DCF = bindates * 0
DCFerr = DCF*0
for myindex in range(len(bindates)):
    udcfinhere = np.where( (deltat > bindates[myindex]) & (deltat <= bindates[myindex]+30))
    print np.shape(udcfinhere)[1]
    #print UDCF[udcfinhere]
    meanarray = np.array([])
    theudcf = UDCF[udcfinhere]
    DCFhere = np.nanmean(UDCF[udcfinhere])
    DCF[myindex] = DCFhere
    #now get the standard error
    thisdiffsq = 0
    for udcfindex in range(len(theudcf)):
        #print udcfindex
        thisdiffsq = (theudcf[udcfindex] - DCFhere)**2
        meanarray = np.append(meanarray, thisdiffsq)
    thisdiffsq =np.sqrt(np.nansum(meanarray))/ (len(theudcf)-1)
    print thisdiffsq
    DCFerr[myindex] = thisdiffsq
#print bindates, DCF    
#print DCFerr
good = np.where(np.isfinite(DCFerr) == 1)
plt.errorbar(bindates[good]+30, DCF[good], yerr=DCFerr[good], fmt='o', linestyle='')
plt.plot(bindates, bindates*0)
plt.xlabel(r'$\Delta$ T (days)')
plt.ylabel(r'DCF')
plt.savefig('1222+216_DCF.eps', format='eps')
#plt.xlim(-500,500)
#plt.show()
plt.close()
plt.cla()

plt.errorbar(bindates[good]+15, DCF[good], yerr=DCFerr[good], fmt='o', linestyle='')
plt.plot(bindates, bindates*0)
plt.ylim(-1.,1.)
plt.xlim(-500.0,500.0)
plt.xlabel(r'$\Delta$ T (days)')
plt.ylabel(r'DCF')
#plt.show()
plt.savefig('1222+216_DCF_zoom.eps', format='eps')
plt.close()
plt.cla()
