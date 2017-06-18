import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerLine2D
from get_order_of_magnitude import get_order_of_magnitude as gom
def plot_models_mc(mjdarr, bbb_integratedflux, sigbump, synchrotron_integratedflux, sigsyn, total_integratedflux, sigflux, plotout, name):
   plt.cla()
   #convert the MJD to the year in order to obtain a second axis scale
   yeararr = mjdarr-51544.0
   yeararr = yeararr/365.25    
   yeararr = yeararr+2000.0   
   def tick_function(X):
    V = 2000.00+(X-51544.00)/365.25
    return ["%.2f" % z for z in V]
   normalize = gom(np.median(total_integratedflux))
   fig = plt.figure(15, figsize=(10,7))
   ax1 = fig.add_subplot(111)
   ax2 = ax1.twiny( )
   line2, = ax1.plot(mjdarr, bbb_integratedflux/normalize, color = (0,0,1), marker = 's', ms=6, 
   label='Big Blue Bump Flux', ls='')
   ax1.errorbar(mjdarr, bbb_integratedflux/normalize, yerr=sigbump/normalize,color= (0,0,1), marker = 's', ms=6, ls='')
   line3, = ax1.plot(mjdarr, synchrotron_integratedflux/normalize, color=(1,0,0), marker = 'o', 
   label = 'Synchrotron Flux', ms=6, ls='')
   ax1.errorbar(mjdarr, synchrotron_integratedflux/normalize, yerr= sigsyn/normalize, color=(1,0,0), marker = 'o', ms=6, ls='')
   line1, = ax1.plot(mjdarr, total_integratedflux/normalize, color=(1,0,1), marker= '^', ms=6, 
   label='Total Flux', ls ='')
   ax1.errorbar(mjdarr, total_integratedflux/normalize, yerr = sigflux/normalize, color=(1,0,1), marker= '^', ms=6, ls='') 
   ax1.set_xlabel('MJD', fontsize=18)
   ax1.set_ylim([0.0, np.amax(total_integratedflux/normalize)+np.std(total_integratedflux/normalize)])
   #ax1.legend(handler_map={type(line1): HandlerLine2D(numpoints=1)},loc='upper left')
   ax2.set_xlabel(r'$Year$', fontsize=18)
   tickarr = ax1.get_xticks()
   xtickarr = ax1.get_xticks().tolist()
   xticks = xtickarr
   xticks[0] = ' '
   ax1.set_xticklabels(xticks)
   ax2xticks = tick_function(tickarr)
   ax2xticks[0] = ' '
   ax2.set_xticklabels(ax2xticks)
   ax1.set_ylabel(r'$F \; (10^{'+str(np.fix(np.log10(normalize)).astype(int))+'} \mathrm{erg \; s}^{-1} \mathrm{cm}^{-2})$', fontsize=18) 
   #plt.show()
   plt.savefig(plotout+name+'.png')
   plt.clf()
   plt.close() 
