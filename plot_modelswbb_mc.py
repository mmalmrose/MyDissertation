import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerLine2D
from get_order_of_magnitude import get_order_of_magnitude as gom
def plot_modelswbb_mc(mjdarr, bbb_integratedflux, sigbump, synchrotron_integratedflux, sigsyn, bb_flux, sig_bbflux, total_integratedflux, sigflux, plotout='/home/mmalmrose/Desktop/', name='test'):
   plt.cla()
   #print bb_flux, sig_bbflux
   #convert the MJD to the year in order to obtain a second axis scale
   yeararr = mjdarr-51544.00
   yeararr = yeararr/365.25    
   yeararr = yeararr+2000.00  
  
   print 'yeararr', yeararr 
   def tick_function(X):
    V = 2000.00+((X-51544.00)/365.25)
    return ["%.1f" % z for z in V]
   normalize = gom(np.median(total_integratedflux))
   fig = plt.figure(25, figsize=(10,7))
   ax1 = fig.add_subplot(111)
   ax2 = ax1.twiny()
   line2, = ax1.plot(mjdarr, bbb_integratedflux/normalize, color = (0,0,1), marker = 's', ms=6, label='Big Blue Bump Flux', ls='')
   ax1.errorbar(mjdarr, bbb_integratedflux/normalize, yerr=sigbump/normalize,color= (0,0,1), marker = 's', ms=6, ls='')
   line3, = ax1.plot(mjdarr, synchrotron_integratedflux/normalize, color=(1,0,0), marker = 'o', label = 'Synchrotron Flux', ms=6, ls='')
   ax1.errorbar(mjdarr, synchrotron_integratedflux/normalize, yerr= sigsyn/normalize, color=(1,0,0), marker = 'o', ls='', ms=6)
   line1, = ax1.plot(mjdarr, total_integratedflux/normalize, color=(1,0,1), marker= '^', ms=6,  label='Total Flux', ls ='')
   ax1.errorbar(mjdarr, total_integratedflux/normalize, yerr = sigflux/normalize, color=(1,0,1), marker= '^', ms=6, ls='') 
   line4, = ax1.plot(mjdarr, bb_flux/normalize, color= (0,0,0), marker = 'D', ms=6, label = 'Black Body Flux', ls = '')
   ax1.errorbar(mjdarr, bb_flux/normalize, yerr = sig_bbflux/normalize, color=(0,0,0), marker='D', ms=6, ls = '')
   ax1.set_xlabel(r'MJD', fontsize=18)
   ax1.set_ylim([0.0, np.amax(total_integratedflux/normalize)+np.std(total_integratedflux/normalize)])
   tickarr = ax1.get_xticks()
   ax2tickarr = tickarr
   xtickarr = ax1.get_xticks().tolist()
   xticks = xtickarr
   xticks[0] = ' '
   ax1.set_xticklabels(xticks)
   ax2.set_xlabel(r'Year', fontsize=18)
   #ax1.legend(handler_map={type(line2): HandlerLine2D(numpoints=1)},loc='upper left')
   tickyear = np.arange(2009,2017)
   ax2.set_xticks(tickyear)
   #ax2_xticks[0]=' '
   ax2.set_xlim(2009,2016)
   ax2.plot(yeararr,bbb_integratedflux/normalize, color=(1,1,1), ls='')
   xticks = ax2.get_xticks().tolist()
   xticks[0] = ' '
   ax2.set_xticklabels(xticks)
   #print 'tickarr', xticks, ax2_xticks
   print ax2.get_xticks(), 'top axis ticks'
   tickarr = ax1.get_xticks()
   ax1.set_ylabel(r'$F \; (10^{'+str(np.fix(np.log10(normalize)).astype(int))+'} \mathrm{erg \; s}^{-1} \mathrm{cm}^{-2})$',fontsize=18) 
   myname = plotout+name+'.png'
   print 'myname' , myname
   #plt.show()
   plt.savefig(myname)
   plt.clf()
   plt.close()
