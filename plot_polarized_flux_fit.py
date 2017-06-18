import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from get_order_of_magnitude import get_order_of_magnitude as gom
def plot_polarized_flux_fit( polarflux, spolarflux, polarnu, mjd, plotout,  restnu, oldq, oldu, nuspec, model, name):
    fig = plt.figure(999, figsize=(10,7)) 
    gs = gridspec.GridSpec(2,1, height_ratios=[3,1])
    ax1 = plt.subplot(gs[0])
    nunorm, fluxnorm, modelnorm = gom(polarnu), gom(polarflux), gom(model)
    ax1.errorbar(polarnu/nunorm, polarflux/fluxnorm, yerr = spolarflux/fluxnorm,  fmt='o', color = (0,1,1))
    ax1.plot(restnu/nunorm, nuspec*np.sqrt(oldu**2 + oldq**2)/fluxnorm )
    ax1.errorbar(polarnu/nunorm, polarflux/fluxnorm, yerr =spolarflux/fluxnorm,  fmt='o', color = (0,1,1))
    ax1.plot(polarnu/nunorm, model/fluxnorm, ls='--', color = (0,0,0), lw=3)
    ytick = ax1.get_yticks().tolist()
    xtick = ax1.get_xticks().tolist()
    for i in range(0, len(xtick)):
        xtick[i] = str(xtick[i])+r'$\times \; \mathrm{10}^{'+str(np.log10(gom(nunorm)).astype(int))+'}$'
    ytick[0] = ' '
    ax2 = plt.subplot(gs[1])
    ax1.set_yticklabels(ytick) 
    ax1.set_xticklabels(xtick)
    ax2.set_xlabel(r'$\nu \; \mathrm{(Hz)}$').set_fontsize(20)
    ax1.set_ylabel(r'$F_{\mathrm{p,\nu\;}} (10^{'+str(np.trunc(np.log10(fluxnorm)).astype(int))+r'}\mathrm{erg\;s}^{-1} \mathrm{\;cm}^{-2} \mathrm{\;Hz}^{-1}\;)$').set_fontsize(20)
    ax1.set_title("MJD = "+str(format(mjd, '.2f'))).set_fontsize(20)
    ax2.errorbar(polarnu/nunorm, (polarflux-model)/(model), yerr=spolarflux/model,fmt='o', color = (0,1,1))
    ax2.plot(polarnu/nunorm, polarflux*0, ls='--', lw=3, color = (0,0,0))
    ax2.set_ylabel(r'$\frac{\mathrm{Residual}}{\mathrm{Model}}$').set_fontsize(15)
    xtick = ax2.get_xticks().tolist()
    for i in range(0, len(xtick)):
        xtick[i] = str(xtick[i])+r'$\times \; \mathrm{10}^{'+str(np.log10(gom(nunorm)).astype(int))+'}$'
    ax2.set_xticklabels(xtick)
    plt.savefig(plotout+str(mjd)+name+'_fit2polflux.png')
    #make a histogram of the residuals
    plt.clf()
    plt.close()
    #plt.figure(10, figsize = (10,7))
    #print (polarflux-polarmodel)/(polarmodel)
    #plt.hist( (polarflux-model)/model, 7, facecolor='g')
    #plt.ylabel('Frequency')
    #plt.xlabel(r'$\frac{\mathrm{Residual}}{\mathrm{Model}}$').set_fontsize(15)
    #plt.savefig(plotout+str(mjd)+name+'_fit2polflux_resid_hist.png')
    #plt.clf()
    #plt.close()

