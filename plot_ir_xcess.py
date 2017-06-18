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
#projected_flux_file = '1222+216_synchrotron_wbb_projected.csv'
#measured_flux_file = '1222+216_IRdata.txt'
#projected_flux_file = '1219+285_synchrotron_projected.csv'
#measured_flux_file = '1219+285_IR_measured.txt'
#projected_flux_file = '3C66A_synchrotron_projected.csv'
#measured_flux_file = '3C66A_IRdata.txt'
projected_flux_file = 'BL_LAC_synchrotron_projected.csv'
measured_flux_file = 'BLLAC_IR_measure.txt'
#projected_flux_file = '3C273_synchrotron_wbb_projected.csv'
#measured_flux_file = '3C273_IR_measure.txt'
#projected_flux_file = '1510-089_synchrotron_wbb_projected.csv'
#measured_flux_file = '1510-089_IR_measure.txt'
#projected_flux_file = 'CTA102_synchrotron_wbb_projected.csv'
#measured_flux_file = 'CTA102_IR_measure.txt'
#projected_flux_file = '3C454.3_synchrotron_wbb_projected.csv'
#measured_flux_file = '3C454.3_IR_measure.txt'
#projected_flux_file = '0420-014_synchrotron_wbb_projected.csv'
#measured_flux_file = '0420-014_IR_measure.txt'
#projected_flux_file = 'CTA26_synchrotron_projected.csv'
#measured_flux_file = 'CTA26_IR_measure.txt'
#projected_flux_file = 'OJ248_synchrotron_projected.csv'
#measured_flux_file = 'OJ248_IR_measure.txt'
#projected_flux_file = '3C279_synchrotron_projected.csv'
#measured_flux_file = '3C279_IR_measure.txt'
projected = np.genfromtxt(projected_flux_file, skip_header=1, delimiter=',', dtype=[('mjd', 'f'), ('jproj', 'f'), ('hproj' , 'f'), ('kproj', 'f')])
real = np.genfromtxt(measured_flux_file, skip_header=0, delimiter=',', dtype=[('mjd', 'f'), ('Telescope', 'S6'), ('filter', 'S6'), ('Obtime', 'f'), ('mag','f'), ('smag','f'), ('mags','f'),  ('cmag','f'), ('scmag','f'), ('cmags','f'), ('flux', 'f'), ('sflux', 'f'), ('fluxs', 'f')] )
projected['mjd']=projected['mjd']-50000
kmeasure = np.where(real['filter'] =='K')
hmeasure = np.where(real['filter'] == 'H')
jmeasure = np.where(real['filter'] == 'J')
print real['filter']
interpkvals = np.interp(real['mjd'][kmeasure], projected['mjd'], projected['kproj']*1000.0)
interphvals = np.interp(real['mjd'][hmeasure], projected['mjd'], projected['hproj']*1000.0)
interpjvals = np.interp(real['mjd'][jmeasure], projected['mjd'], projected['jproj']*1000.0)
#print interpvals
diffkvals = real['flux'][kmeasure] - interpkvals - 6.84
diffhvals = real['flux'][hmeasure] - interphvals - 9.08
diffjvals = real['flux'][jmeasure] - interpjvals - 7.38
diffkerr = np.sqrt(real['fluxs'][kmeasure]**2 + np.std(projected['kproj']*1000)**2)
diffherr = np.sqrt(real['fluxs'][hmeasure]**2 + np.std(projected['hproj']*1000)**2)
diffjerr = np.sqrt(real['fluxs'][jmeasure]**2 + np.std(projected['jproj']*1000)**2)
#print np.mean(diffkvals),np.mean(differr), 'average k diff'
#print real['flux'][kmeasure]-interpvals
#plt.plot(real['mjd'][kmeasure], real['flux'][kmeasure], 'o', color =(1,0,0), markersize=15)
#plt.plot(projected['mjd'], projected['kproj']*1000.0, marker='s', color = (1,0,0), ls='none')
#plt.errorbar(real['mjd'][kmeasure], interpkvals, np.std(projected['kproj']*1000), marker='s', color='b', ls='none' )
#plt.xlabel('MJD')
#plt.ylabel(r'K-band Flux (mJy)')
#plt.show()
#plt.close()
#plt.errorbar(real['mjd'][kmeasure], diffvals, differr, marker='o', ls='none')
#plt.xlabel('MJD')
#plt.ylabel(r'K-band Flux (mJy)')
#plt.show()
#plt.close()

#plt.plot(real['mjd'][hmeasure], real['flux'][hmeasure], 'o', color =(1,0,0), markersize=15)
#plt.plot(projected['mjd'], projected['hproj']*1000.0, marker='s', color = (1,0,0), ls='none')
#plt.plot(real['mjd'][hmeasure], interphvals, marker='s', color = 'b')
#plt.show()
#plt.close()
#plt.plot(real['mjd'][hmeasure],  real['flux'][hmeasure] - interphvals, marker='', color = 'b', ls = 'none')
#plt.show()
#plt.close()
#plt.plot(real['mjd'][jmeasure], real['flux'][jmeasure], 'o', color =(1,0,0), markersize=15)
#plt.plot(real['mjd'][jmeasure], interpjvals, marker='s', color= 'b')
#plt.plot(projected['mjd'], projected['jproj']*1000.0, marker='s', color = (1,0,0), ls='none')
#plt.show()
#plt.close()



#make a pretty plot

gs = gridspec.GridSpec(3,1, height_ratios=[1,1,1])
ax1 = plt.subplot(gs[0])
ax1.plot(real['mjd'][kmeasure], real['flux'][kmeasure], 'o', color =(0,0.5,0.5), markersize=15)
ax1.plot(projected['mjd'], projected['kproj']*1000.0, marker='s', color = (1,0,0), ls='none')
ax1.errorbar(real['mjd'][kmeasure], interpkvals, np.std(projected['kproj']*1000), marker='*', color='b', ls='none' , markersize=15)
ax1.set_ylabel(r'$F_K$ (mJy)')
ax2 = plt.subplot(gs[1])
ax2.plot(real['mjd'][hmeasure], real['flux'][hmeasure], 'o', color =(0,0.5,0.5), markersize=15)
ax2.plot(projected['mjd'], projected['hproj']*1000.0, marker='s', color = (1,0,0), ls='none')
ax2.plot(real['mjd'][hmeasure], interphvals, marker='*', color = 'b', markersize=15, ls='none')
ax2.errorbar(real['mjd'][hmeasure], interphvals, np.std(projected['hproj']*1000), marker='*', color='b', ls='none' , markersize=15)
ax2.set_ylabel(r'$F_H$ (mJy)')
ax3 = plt.subplot(gs[2])
ax3.plot(real['mjd'][jmeasure], real['flux'][jmeasure], 'o', color =(0,0.5,0.5), markersize=15)
ax3.plot(real['mjd'][jmeasure], interpjvals, marker='*', color= 'b', markersize=15, ls='none')
ax3.plot(projected['mjd'], projected['jproj']*1000.0, marker='s', color = (1,0,0), ls='none')
ax3.errorbar(real['mjd'][jmeasure], interpjvals, np.std(projected['jproj']*1000), marker='*', color='b', ls='none' , markersize=15)
ax3.set_ylabel(r'$F_J$ (mJy)')
ax3.set_xlabel(r'MJD')
plt.savefig('IR_Measured_and_Projected_Flux.eps', format='eps')#, dpi='1000')
plt.close()

gs = gridspec.GridSpec(3,1, height_ratios=[1,1,1])
ax1 = plt.subplot(gs[0])
ax1.plot(real['mjd'][kmeasure], diffkvals, 'o', color =(0,0.5,0.5), markersize=15)
plt.errorbar(real['mjd'][kmeasure], diffkvals, diffkerr, marker='o', ls='none', color =(0,0.5,0.5), markersize=15)
ax1.set_ylabel(r'$\Delta F_K$ (mJy)')
ax2 = plt.subplot(gs[1])
ax2.plot(real['mjd'][hmeasure], diffhvals, 'o', color =(0,0.5,0.5), markersize=15)
plt.errorbar(real['mjd'][hmeasure], diffhvals, diffherr, marker='o', ls='none', color =(0,0.5,0.5), markersize=15)
ax2.set_ylabel(r'$\Delta F_H$ (mJy)')
ax3 = plt.subplot(gs[2])
ax3.plot(real['mjd'][jmeasure], diffjvals, 'o', color =(0,0.5,0.5), markersize=15)
plt.errorbar(real['mjd'][jmeasure], diffjvals, diffjerr, marker='o', ls='none', color =(0,0.5,0.5), markersize=15)
ax3.set_ylabel(r'$\Delta F_J$ (mJy)')
ax3.set_xlabel(r'MJD')
plt.savefig('IR_excess.eps', format='eps')#, dpi='1000')
plt.close()

#get a blackbody curve at various T in order to compare to the observed IR excess
bbnu = np.linspace(0.1,7, num=1000)*1.0e+14
print bbnu
bb1200 = bb(bbnu, 1200)
bb1800 = bb(bbnu, 1800)


average_excess=np.array([np.mean(diffjvals), np.mean(diffhvals), np.mean(diffkvals)])
ir_lambda = np.array([1.2, 1.6,2.2])/1.941
ir_nu = 3.0e+14/ir_lambda
print ir_nu, 'irnu'

plt.plot(ir_nu, ir_nu*1.0e-26*average_excess, 'o', color=(0,0,1))
plt.plot(bbnu, 8.0e-11*bbnu*bb1200/np.max(bb1200*bbnu), ls='--', color = (1,0,0))
plt.plot(bbnu, 8.0e-11*bbnu*bb1800/np.max(bb1800*bbnu), ls='-.', color = (0,0,1))
plt.errorbar(ir_nu, 1.0e-26*ir_nu*average_excess, [np.mean(diffjerr)/np.sqrt(len(diffjerr)), np.mean(diffherr)/np.sqrt(len(diffherr)), np.mean(diffkerr)/np.sqrt(len(diffkerr))]*ir_nu*1.0e-26, ls='none', color=(0,0,1))
#plt.plot(bbnu/1.0e+15, bbnu*bb1200)
#plt.ylim(0,2)
#plt.xlim(0,5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\nu \/\mathrm{(Hz)}$')
plt.ylabel(r'$\nu F_{\nu} \/\mathrm{( erg \/s^{-1} cm^{2})}$')
plt.ylim(1.0e-12,1.0e-9)
plt.savefig('Average_IR_Excess_SED.eps', format='eps')#, dpi='1000')
plt.close()
plt.plot(bbnu, bbnu*bb1200)
plt.xscale('log')
plt.yscale('log')
plt.ylim(1.0, 1.0e+7)
#plt.show()
print 'test', np.std(diffkvals), np.mean(diffkvals), np.mean(diffkerr)
#print a crude estimate of the ratio of variability to the uncertainty
jvar= np.std(diffjvals)/np.mean(diffjvals)
hvar =np.std(diffhvals)/np.mean(diffhvals)
kvar = np.std(diffkvals)/np.mean(diffkvals)
print kvar
#print 'jvar, hvar, kvar, jmean, hmean, kmean, jerr, herr, kerr', jvar, hvar, kvar, np.mean(diffjvals), np.mean(diffhvals), np.mean(diffkvals), np.mean(diffjerr)/np.mean(diffjvals), np.mean(diffherr)/np.mean(diffhvals), np.mean(diffkerr)/np.mean(diffkvals)
 
print 'j, deltaj, sigj/j, h, deltah, sigh/h, k, deltak, sigk/k',  np.mean(diffjvals), jvar, np.mean(diffjerr)/np.mean(diffjvals), np.mean(diffhvals), hvar,np.mean(diffherr)/np.mean(diffhvals), np.mean(diffkvals), kvar,  np.mean(diffkerr)/np.mean(diffkvals)
print 'divided by sigma', jvar/np.mean(diffjerr) , hvar/np.mean(diffherr), kvar/ np.mean(diffkerr)
print 'k fk', ir_nu[2] * np.mean(diffkvals)
