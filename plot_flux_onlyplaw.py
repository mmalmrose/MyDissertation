import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from get_order_of_magnitude import get_order_of_magnitude as gom
def plot_flux_onlyplaw(total_integratedflux, onlyplaw_alpha, mjdarr, oldplotout, name):
    inverseflux = np.float64(1.0/total_integratedflux)
    #get a linear fit to the inverse flux
    #group the arrays by mjd so that the plot can be color coded.
    bluegroup = np.where(mjdarr < 55450)
    redgroup = np.where( np.logical_and( mjdarr >55450, mjdarr < 55800))
    greengroup = np.where(np.logical_and( mjdarr >55800, mjdarr < 56200))
    brgroup = np.where(np.logical_and( mjdarr >56200, mjdarr <56500))
    rggroup = np.where(np.logical_and( mjdarr > 56500, mjdarr < 56717) )
    blackgroup = np.where(mjdarr > 56717)
    normalinverse = gom(inverseflux)
    normalized = inverseflux/normalinverse
    linexarr = np.linspace(0.01,7, num=400)
    linefit = np.polyfit(normalized, onlyplaw_alpha, 1)
    theline = linefit[0]*linexarr + linefit[1]
    # If the spectral index goes to -2, this would indicate a very hot 
    #Blackbody.  Anything with a steeper spectral index will not be thermal
    #So figure out what the flux of a alpha=-2 power law is with this regression
    minbump = np.float64(linefit[0]/( (-2.0) - linefit[1]))
    minbump= (minbump/normalinverse)
    print linefit, 'linefit'
    fig = plt.figure(22, figsize=(10,7))
    x = (1.0/(inverseflux/normalinverse))
    line1, = plt.plot(x[bluegroup], onlyplaw_alpha[bluegroup], marker='o', 
    color = (0,0, 1), linestyle='' , label='MJD < 55450', ms=10)
    line2, = plt.plot(x[redgroup],  onlyplaw_alpha[redgroup], marker='<',
    color = (1,0, 0), linestyle='' ,label='55450 < MJD < 55800', ms=10)
    line3, = plt.plot(x[greengroup],  onlyplaw_alpha[greengroup], marker='>', 
    color = (0,1, 0),linestyle='' , label='55800 < MJD < 56200', ms=10)
    line4, = plt.plot(x[brgroup],  onlyplaw_alpha[brgroup], marker='H', 
    color = (1,0, 1), linestyle='' ,label = '56200 < MJD < 56500', ms=10)    
    line5, = plt.plot(x[rggroup],  onlyplaw_alpha[rggroup], marker='D', 
    color = (1,1, 0), linestyle='' ,label = '56500< MJD < 56714')    
    line6, = plt.plot(x[blackgroup],  onlyplaw_alpha[blackgroup], marker='s', 
    color = (0,0, 0),linestyle='' , label = 'MJD > 56714', ms=10) 
    plt.plot(1.0/linexarr, theline, color=(0,0,1), lw=4)
    plt.xlim([0.0, 3.0])
    plt.xlabel(r'$\mathrm{Flux} \; (10^{-'+str(np.trunc(np.log10(normalinverse)
    ).astype(int))+r'} \; \mathrm{erg} \; \mathrm{s}^{-1} \;\mathrm{cm}^{-2})$' )
    plt.ylabel(r'$\alpha_{F_{\nu}}$')
    plt.ylim([-2.0, 1.0])
    #plt.legend(loc ='lower right')
    plt.legend(handler_map={type(line1): HandlerLine2D(numpoints=1)}, loc='lower right')
    pp = oldplotout+name+'.png'
    plt.savefig(pp, padinches=0.85)
    plt.clf()
    plt.close() 
    fig2 = plt.figure(23, figsize=(10,7))
    x = (inverseflux/normalinverse)
    line1, = plt.plot(x[bluegroup], onlyplaw_alpha[bluegroup], marker='o', 
    color = (0,0, 1), linestyle='' , label='MJD < 55450', ms=10)
    line2, = plt.plot(x[redgroup],  onlyplaw_alpha[redgroup], marker='<',
    color = (1,0, 0), linestyle='' ,label='55450 < MJD < 55800', ms=10)
    line3, = plt.plot(x[greengroup],  onlyplaw_alpha[greengroup], marker='>', 
    color = (0,1, 0),linestyle='' , label='55800 < MJD < 56200', ms=10)
    line4, = plt.plot(x[brgroup],  onlyplaw_alpha[brgroup], marker='H', 
    color = (1,0, 1), linestyle='' ,label = '56200 < MJD < 56500', ms=10)    
    line5, = plt.plot(x[rggroup],  onlyplaw_alpha[rggroup], marker='D', 
    color = (1,1, 0), linestyle='' ,label = '56500< MJD < 56714')    
    line6, = plt.plot(x[blackgroup],  onlyplaw_alpha[blackgroup], marker='s', 
    color = (0,0, 0),linestyle='' , label = 'MJD > 56714', ms=10) 
    plt.plot(linexarr, theline, color=(0,0,1), lw=4)
    plt.xlim([0.0, 5.0])
    plt.ylim([-2.0, 1.0])
    plt.xlabel(r'$\mathrm{Inverse\;Flux} \; (10^{-'+str(np.trunc(np.log10(normalinverse)
    ).astype(int))+r'} \; \mathrm{erg} \; \mathrm{s}^{-1} \;\mathrm{cm}^{-2})^{-1}$' )
    plt.ylabel(r'$\alpha_{F_{\nu}}$')
    #plt.legend(loc ='lower left')
    plt.legend(handler_map={type(line1): HandlerLine2D(numpoints=1)}, loc='lower left')
    pp = oldplotout+name+'_linear.png'
    plt.savefig(pp, padinches=0.85)
    plt.clf()
    plt.close()
    return minbump
