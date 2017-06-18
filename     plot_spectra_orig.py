def plot_spectra_orig(origcorrected, thisspec, nuorig, nuspec, restwave, restnu):
    import numpy as np
    import scipy as sp
    import matplotlib    
    import matplotlib.pyplot as plt
    #pol = np.sqrt(qspec**2 + uspec**2)
    plt.figure(2, figsize=(10,8))
    plt.subplot(311)
    plt.plot(wave, qspec)
    plt.ylabel(r'$Q/I$')
    plt.title('MJD = '+str(qmjd))
    plt.subplot(312)
    plt.plot(wave, uspec)
    plt.ylabel(r'$U/I$')
    plt.title=(str(
    plt.subplot(313)
    plt.plot(wave, pol)
    plt.xlabel(r'$\lambda \;(\mathrm{\AA})$')
    plt.ylabel(r'$p$') 
    plt.savefig(dirout+str(qmjd)+'_q_u_p.eps', padinches=0.2) 
    plt.clf()
    plt.figure(3, figsize=(10,8))
    plt.subplot(311)
    plt.plot(wave, qspec)
    plt.ylabel(r'$Q/I$')
    plt.title('MJD = '+str(qmjd))
    plt.subplot(312)
    plt.plot(wave, uspec)
    plt.ylabel(r'$U/I$')
    plt.subplot(313)
    plt.plot(wave, theta)
    plt.xlabel(r'$\lambda \;(\mathrm{\AA})$')
    plt.ylabel(r'$\theta$') 
    plt.savefig(dirout+str(qmjd)+'_q_u_theta.eps', padinches=0.2)
    plt.clf() 