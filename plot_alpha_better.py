def plot_alpha_better(mjdarr, alpha_array, alpha_err, plotout, onlyplaw_alpha, sigma_onlyplaw_alpha, name):
    import numpy as np
    import scipy as sp
    import matplotlib    
    import matplotlib.pyplot as plt
    #convert the MJD to the year in order to obtain a second axis scale
    yeararr = mjdarr-51544.0
    yeararr = yeararr/365.25
    yeararr = yeararr+2000.0
    #print flagbbb
    #print bbb_integratedflux
    def tick_function(X):
        V = 2000.0+(X-51544.0)/365.25
        return ["%4d" % z for z in V]
    fig = plt.figure(6, figsize=(10,7))
    #ax1 = fig.add_subplot(111)
    ax1 = plt.subplot2grid((3,10), (0,0), rowspan=3, colspan=7)
    ax2 = ax1.twiny( )
    plt.plot(mjdarr, alpha_array,label = r'$\alpha-F_p$', marker='o', color = (0,0,1), markersize=10, linestyle = '')
    ax1.errorbar(mjdarr, alpha_array, yerr=alpha_err, color= (0,0,1), marker='o' , markersize=10, linestyle='')
    plt.plot(mjdarr, onlyplaw_alpha, label = r'$\alpha-F_{\nu}$ ', marker='s', linestyle = '', color = (0,0,0))
    ax1.errorbar(mjdarr, onlyplaw_alpha, yerr=sigma_onlyplaw_alpha, color = (0,0,0), fmt='s', markersize=10)
    ax1.set_xlabel(r'$MJD $')
    ax1.set_ylabel(r'$\alpha$' )
    ax1.set_ylim([np.min(np.append(alpha_array,onlyplaw_alpha))-np.std(onlyplaw_alpha),np.max(np.append(alpha_array, onlyplaw_alpha))+np.std(onlyplaw_alpha)])
    ax1.legend(loc='upper left')
    ax2.set_xlabel(r'$Year$')
    tickarr = ax1.get_xticks()
    xtickarr = ax1.get_xticks().tolist()
    xticks = xtickarr
    xticks[0] = ' '
    ax1.set_xticklabels(xticks)
    ax2xticks = tick_function(tickarr)
    ax2xticks[0] = ' '
    ax2.set_xticklabels(ax2xticks)
    fig.subplots_adjust(top=0.85)
    pp = plotout+name+'_spectral_index.png'
    ax4 = plt.subplot2grid((3,10), (0,7), rowspan=3, colspan=3)
    ax4.hist([alpha_array], 6, color = (0,0,1), orientation='horizontal', hatch='/')
    ax4.hist([onlyplaw_alpha], 6, color=(0,0,0), orientation='horizontal', hatch='x')
    ax4.set_xlabel(r'Frequency')
    ax4.set_ylabel(r'$\alpha$')
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position('right')
    ax4.xaxis.set_label_position('top')
    xtickarr = ax4.get_xticks().tolist()
    aaa = len(xtickarr)
    step = np.fix((xtickarr[aaa-1]-xtickarr[0])/3.0)
  #  ax4.xaxis.set_ticks(np.arange(xtickarr[0], xtickarr[aaa-1], step))
  #  ax4.set_ylim([np.min(np.append(alpha_array,onlyplaw_alpha))-np.std(alpha_array),np.max(np.append(alpha_array, onlyplaw_alpha))+np.std(alpha_array)])
    xtickarr = ax4.get_xticks().tolist()
    xtickarr[0]=' '
   #ax4.set_xticklabels(xtickarr)
    plt.savefig(pp, padinches=0.85)
    plt.clf()
    plt.close(fig)
    
    fig = plt.figure(6, figsize=(10,7))
    ax1 = plt.subplot2grid((3,10), (0,0), rowspan=3, colspan=7)
    ax2 = ax1.twiny( )
    newerror = np.sqrt(alpha_err**2 + sigma_onlyplaw_alpha**2)
    plt.plot(mjdarr, alpha_array- onlyplaw_alpha,label = r'$\alpha-F_p - \alpha-F$', marker='o', color = (0,0,1), markersize=10, linestyle = '')
    ax1.errorbar(mjdarr, alpha_array- onlyplaw_alpha, yerr=alpha_err, color= (0,0,1), marker='o' , markersize=10, linestyle='')
    ax1.set_xlabel(r'$MJD $')
    ax1.set_ylabel(r'$\Delta \alpha$' )
    ax1.set_ylim([-1,1])
    ax2.set_xlabel(r'$Year$')
    tickarr = ax1.get_xticks()
    xtickarr = ax1.get_xticks().tolist()
    xticks = xtickarr
    xticks[0] = ' '
    ax1.set_xticklabels(xticks)
    ax2xticks = tick_function(tickarr)
    ax2xticks[0] = ' '
    ax2.set_xticklabels(ax2xticks)
    ax4 = plt.subplot2grid((3,10), (0,7), rowspan=3, colspan=3)
    ax4.hist(alpha_array- onlyplaw_alpha, 6, color = (0,0,1), orientation='horizontal')
    ax4.set_xlabel(r'Frequency')
    ax4.set_ylabel(r'$\alpha$')
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position('right')
    ax4.xaxis.set_label_position('top')
    xtickarr = ax4.get_xticks().tolist()
    aaa = len(xtickarr)
    step = np.fix((xtickarr[aaa-1]-xtickarr[0])/5.0)
   # ax4.xaxis.set_ticks(np.arange(xtickarr[0], xtickarr[aaa-1], step))
    #ax4.set_ylim([np.min(alpha_array-onlyplaw_alpha)-np.std(alpha_array),np.max(alpha_array-onlyplaw_alpha)+(2*np.std(alpha_array))])
    ax4.set_ylim([-1,1])
    xtickarr = ax4.get_xticks().tolist()
    xtickarr[0]=' '
    ax4.set_xticklabels(xtickarr)
    fig.subplots_adjust(top=0.85)
    pp = plotout+name+'delta_spectral_index.png'
    plt.savefig(pp, padinches=0.85)
    plt.clf()
    plt.close(fig)
