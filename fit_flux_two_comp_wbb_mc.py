import lmfit
from lmfit import  Model, minimize, Parameters
from plot_polarized_flux_fit import  plot_polarized_flux_fit
from find_plaw_resid import find_plaw_resid
from find_plaw import find_plaw
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from find_line_resid import find_line_resid
from find_line import find_line
from find_modified_plaw import find_modified_plaw
from get_order_of_magnitude import get_order_of_magnitude as gom
from find_modified_plaw_resid import find_modified_plaw_resid
from twocomponentmodel_resid import twocomponentmodel_resid
from twocomponentmodel import  twocomponentmodel
from get_synchrotron import get_synchrotron 
from get_bigbluebump import get_bigbluebump
from chi_square import chi_square
from blackbody import blackbody as bb
from scipy.optimize import curve_fit
from twocomponentmodel_BB_resid import twocomponentmodel_BB_resid
from twocomponentmodel_BB import twocomponentmodel_BB
from rebin import rebin
def fit_flux_two_comp_wbb_mc(restnu, polarnu, polarflux, spflux, fitwindow, normflux, plotout, qmjd, oldq, oldu, nuorig, fitflux, plaw_params, normalflux, normnu, sfitflux, twocomp_params, nuspec, name, just1plaw_model):
    fitspectrum = fitflux*0.0
    synarr = np.array([])
    bumparr= np.array([])
    alpharr = np.array([])
    modelfluxarr = np.array([])
    modelsynarr = np.array([])
    modelbumparr = np.array([])
    modelbbodyarr = np.array([]) 
    maxiter = 250
    fitflux = fitflux#-polarflux
    for myi in range(0, maxiter):
        randomarray = np.random.randn(len(fitflux))
        #print randomarray[2]
        fitspectrum[0:] = fitflux[0:]+(randomarray[0:]*sfitflux[0:])
        if myi == (maxiter -1):
            fitspectrum = fitflux
        #plt.plot(polarnu, fitspectrum)    
        #amin = plaw_params['alpha'].value-(1.0*plaw_params['alpha'].stderr)
        #plaw_params['alpha'].value =  plaw_params['alpha'].value -0.05
        amin = plaw_params['alpha'].value-(0.1)
        amin = plaw_params['alpha'].value-(1.0)
        #amax = plaw_params['alpha'].value+ (1.0*plaw_params['alpha'].stderr)
        amax = plaw_params['alpha'].value+ (0.1)
        amax = plaw_params['alpha'].value+ (1.0)
        minsyn = plaw_params['norm'].value*normalflux
        abegin = (np.mean(fitflux/normalflux)/2)*np.mean(polarnu/normnu)**((-1.0)*plaw_params['alpha'].value)
        abegin = (np.mean(fitflux/normalflux)/2.0)*np.mean(polarnu/normnu)**((-1.0)*plaw_params['alpha'].value)
        twocomp_BB_params=Parameters()
        twocomp_BB_params.add('synnormal', value =abegin, min =minsyn)
        #twocomp_BB_params.add('synalpha', value = plaw_params['alpha'].value , min= amin, max = amax)
        twocomp_BB_params.add('synalpha', value = plaw_params['alpha'].value, vary=False)
        twocomp_BB_params.add('bumpindex', value = np.float64(-1.0/3.0), vary=False)
       # twocomp_BB_params.add('bumpindex', value = np.float64(-0.46), vary=False)
        twocomp_BB_params.add('temperature', value=np.float64(12000.0), min=5000., max=60000)
        #twocomp_BB_params.add('temperature', value=np.float64(20000.0), min = 5000, max = 50000)
        twocomp_BB_params.add('blackbody_norm', value = 0.2, min=np.float64(0.0), max=100)
        #twocomp_BB_params.add('blackbody_norm', value = 20.0, vary=False)
        blackbody = bb(polarnu, twocomp_BB_params['temperature'].value)
        blackbodymodel = bb(restnu, twocomp_BB_params['temperature'].value)
        blackbodymodel = blackbodymodel/np.max(blackbodymodel)
        blackbody = blackbody/np.max(blackbody) 
        #bbegin = (np.mean(fitflux/normalflux)/3.)*np.mean(polarnu/normnu)**(-1./3.)
        bbegin = 0.145 #1222+216
        #bbegin = 0.03 #CTA26
        #bbegin = 0.1 #BLLAC 
        bbegin=  0.0 #3C66A and other BL Lacs
        #bbegin = .06 #3C273
        #bbegin = 0.04 #CTA102
        #bbegin = 0.005 # 3C454.3
        minbumpflux = np.float64(1.84644587168e-12)
        integrated = np.max(restnu)**((-1)*twocomp_params['bumpindex'].value
        +1.0) - np.min(restnu)**((-1)*twocomp_params['bumpindex'].value+1.0)  
        minbumpnorm =  minbumpflux *((-1)*twocomp_params['bumpindex'].value+1.0
        )*np.power(normnu, (-1)*twocomp_params['bumpindex'].value)/integrated
        minbumpnorm = minbumpnorm/normalflux
        aaa= 2.808E-17/(np.power(normnu, (0.77))*normflux)
        aaa = 8*1.80505565259e-23/(normalflux*np.power(normnu, (1./3.)))
        minbumpnorm = aaa/2.0
        #print 'aaa', aaa
        #if np.absolute(qmjd - 56714.0) < 1.0:
          #  aaa= 2.808E-17/(np.power(normnu, (0.77))*normflux)
            #aaa = 0.46601200360 
            #aaa = 8*1.80505565259e-23/(normalflux*np.power(normnu, (1./3.)))

         #   print 'aaa', aaa
            #twocomp_BB_params.add('bumpnormal', value=aaa, vary=False)
            #twocomp_BB_params.add('bumpnormal', value = aaa, min=minbumpnorm/5.0)
            #twocomp_BB_params.add('bumpnormal', value = aaa, min=0.0)
        twocomp_BB_params.add('bumpnormal', value = bbegin, vary=False)
        #twocomp_BB_params.add('bumpnormal', value = bbegin, min=0.0, vary = True)
        #else:
           # twocomp_BB_params.add('bumpnormal', value = aaa, min=minbumpnorm/5.0)
         #   twocomp_BB_params.add('bumpnormal', value = aaa, min=0.0)     
            #twocomp_BB_params.add('bumpnormal', value=aaa, vary=False)
    #Now solve with Levenberg-Marquadt
    #print twocomp_BB_params , 'now with blackbody'
    #print normalflux
    #print twocomp_BB_params['temperature'].value, 'temperature'
    #model_for_spectrum_bb = lmfit.minimize(twocomponentmodel_BB_resid, 
    #twocomp_BB_params, args=((polarnu[fitwindow]/normnu), normnu, (fitflux[fitwindow]/gom(nuspec)), 
    #sfitflux[fitwindow]/gom(nuspec) ), method='Nelder') 
        bumpmodelpolnu =  get_bigbluebump(twocomp_BB_params, polarnu/normnu)*normalflux  
        model_for_spectrum_bb = minimize(twocomponentmodel_BB_resid, twocomp_BB_params, 
        args=(polarnu[fitwindow]/normnu, normnu, (fitspectrum[fitwindow]/normalflux), 
        sfitflux[fitwindow]/gom(nuspec)))
    #model_for_spectrum_bb = curve_fit(twocomponentmodel_BB, polarnu[fitwindow]/normnu, normnu, (fitflux[fitwindow]/gom(nuspec)), p0=model_for_spectrum_bb, sigma =  sfitflux[fitwindow]/gom(nuspec))
    #get initial values for parameters with  Nelder-Mead
    #print fitflux/normalflux
    #print blackbody
    #get the blackbody flux
        blackbodymodel =  bb(restnu, twocomp_BB_params['temperature'].value)* twocomp_BB_params['blackbody_norm'].value*normalflux
        bbmodelpolnu =  bb(polarnu, twocomp_BB_params['temperature'].value)* twocomp_BB_params['blackbody_norm'].value*normalflux
       # blackbodymodel =  bb(restnu, twocomp_BB_params['temperature'].value)* twocomp_BB_params['blackbody_norm'].value*1.0e-26
       # bbmodelpolnu =  bb(polarnu, twocomp_BB_params['temperature'].value)* twocomp_BB_params['blackbody_norm'].value*1.0e-26
    #plt.plot(restnu, blackbodymodel)
    #plt.show()
        bumpmodelwbb = get_bigbluebump(twocomp_BB_params, restnu/normnu)*normalflux
        bumpmodelpolnu =  get_bigbluebump(twocomp_BB_params, polarnu/normnu)*normalflux
        synmodelwbb =  get_synchrotron(twocomp_BB_params, restnu/normnu)*normalflux#+rebin(polarflux, restnu.shape)
        synmodelpolnu = get_synchrotron(twocomp_BB_params, polarnu/normnu)*normalflux#+ polarflux
        modelwbb=( blackbodymodel+synmodelwbb + bumpmodelwbb)/normalflux #+rebin(polarflux, restnu.shape)
        modelfluxarr = np.append(modelfluxarr, np.trapz(bumpmodelwbb+synmodelwbb+blackbodymodel, restnu)*(-1))#+ np.trapz(polarflux, polarnu)*(-1.0))
        modelsynarr =  np.append(modelsynarr, np.trapz(synmodelwbb, restnu)*(-1.0))#+ np.trapz(polarflux, polarnu)*(-1.0)) 
        modelbumparr = np.append(modelbumparr, np.trapz(bumpmodelwbb, restnu)*(-1))
        #print synfluxmodel, 'synfluxmodel'
        modelbbodyarr = np.append(modelbbodyarr, np.trapz(blackbodymodel, restnu)*(-1)) 
        #print modelwbb 
        modelwbbscale = bb(restnu, twocomp_BB_params['temperature'].value)
        modelwbbscale = modelwbbscale/normalflux
        mymodel = (synmodelpolnu+ bumpmodelpolnu+bbmodelpolnu)  
        mymodel = mymodel/normalflux
        mmodel = synmodelpolnu + bumpmodelpolnu + bbmodelpolnu
    fig=plt.figure(33, figsize=(10,7))
    gs = gridspec.GridSpec(2,1, height_ratios=[3,1])
    ax1 = plt.subplot(gs[0])
    #print len(just1plaw_model), 'len just1plaw_model'
    #print len(restnu), 'len restnu'
    fitflux = fitflux#+polarflux
    ax1.set_title("MJD = "+str(format(qmjd, '.2f'))).set_fontsize(18)
    ax1.plot(restnu/normnu, nuspec/normalflux, color = (0,0,1))
    ax1.plot(restnu/normnu, synmodelwbb/normalflux, ls = ':', color = (1,0,0), lw=3)
    ax1.plot(restnu/normnu, bumpmodelwbb/normalflux, ls='--', color = (0,0,1), lw=3)
    ax1.plot(restnu/normnu, blackbodymodel/normalflux, ls = '-.', color=(0,0,0), lw=3)
    ax1.plot(polarnu/normnu,just1plaw_model/normalflux, color = (1, 0, 1), lw=4)
    ax1.plot(polarnu/normnu, fitflux/normalflux, marker='o', ls='', color = (1,0,0))
    ax1.plot(polarnu[fitwindow]/normnu, fitflux[fitwindow]/normalflux, marker='s', color=(0,1,1), ms=12, ls='')
    ax1.errorbar(polarnu/normnu, fitflux/normalflux, yerr=sfitflux/normalflux, color = (1,0,0), marker='o', ls='')
    xtick = ax1.get_xticks().tolist()
    for i in range(0, len(xtick)):
        xtick[i] = str(xtick[i])+r'$\times \; \mathrm{10}^{'+str(np.log10(gom(normnu)).astype(int))+'}$'
    ax1.set_xticklabels(xtick)
    #ax1.set_xlabel(r'$\nu\; $(10$^{14} \;$Hz)').set_fontsize(15)
    ax1.set_ylabel(r'$F_{\mathrm{\nu\;}} (10^{'+str(np.trunc(np.log10(normalflux)).astype(int))+r'}\mathrm{erg} \; \mathrm{s}^{-1} \; \mathrm{cm}^{-2} \mathrm{\;Hz}^{-1}\;)$').set_fontsize(15)
    ax1.plot(restnu/normnu, modelwbb, ls='--', color = (0.5, 0, 0.5), lw=4)
    ax2 = plt.subplot(gs[1])
    resid = ((nuspec/normalflux) - modelwbb)/modelwbb
    resid2 = (fitflux - mymodel)/mymodel
    ax2.plot(restnu/normnu, resid)
    #print mymodel

    ax2.plot(polarnu/normnu, (fitflux - mmodel)/mmodel, marker='o', color = (1,0,0), ls='')
    ax2.plot(polarnu[fitwindow]/normnu, (fitflux[fitwindow] -mmodel[fitwindow])/mmodel[fitwindow], marker='s', color = (0,1,1), ls='', ms=12) 
    ax2.errorbar(polarnu[fitwindow]/normnu, (fitflux[fitwindow] -mmodel[fitwindow])/mmodel[fitwindow], yerr  =sfitflux[fitwindow]/mmodel[fitwindow], color = (1,0,0), ls='', marker = 'o')
    ax2.plot(restnu/normnu, restnu*0, color = (0,0,0), lw=3, ls='--')
    xtick = ax2.get_xticks().tolist()
    for i in range(0, len(xtick)):
        xtick[i] = str(xtick[i])+r'$\times \; \mathrm{10}^{'+str(np.log10(gom(normnu)).astype(int))+'}$'
    ax2.set_xticklabels(xtick)  
    ax2.set_ylabel(r'$ \frac{\mathrm{Residual}}{\mathrm{Model}} $')
    ax2.set_xlabel(r'$\nu \; \mathrm{(Hz)}$').set_fontsize(15)
    #plt.plot(restnu, modelwbbscale)
    #print plotout, 'plotout'
    plt.savefig(plotout+name+'two_comp_with_BB.png', padinches=2)
    plt.clf()
    plt.close()
    #plt.plot(restnu, modelwbbscale)
    #plt.show()
    #print synmodelwbb, 'synmodelwbb'
   
    #Get the chi square for this particular model
    scfitflux = fitflux/normalflux
    scsfitflux = sfitflux/normalflux
    #print mmodel
    mychisquarewbb = chi_square(mmodel[fitwindow], fitflux[fitwindow], sfitflux[fitwindow])
    print 'mychisquarewbb', mychisquarewbb, 'mychisquarewbb'
    #print ( (modelwbb[fitwindow]-scfitflux[fitwindow])**2)
    #print np.shape(fitwindow), 'shape of window'
    dof = len(fitwindow[0]) - 4.0
    print 'dof', dof
    #print "len(fitwindow) - 4", len(fitwindow) - 4.0
    #print 'modelwbb[fitwindow], fitflux[fitwindow], sfitflux[fitwindow]', modelwbb[fitwindow], fitflux[fitwindow]/gom(nuspec), sfitflux[fitwindow]/gom(nuspec)
    mychisquarewbb = mychisquarewbb/dof
    print 'mychisquarewbb', mychisquarewbb
    bbbumpfluxmodel, blackbodyfluxmodel, synfluxmodel  =   modelbumparr[maxiter-1],  modelbbodyarr[maxiter-1], modelsynarr[maxiter-1]
    sigbbbumpfluxmodel, sigblackbodyfluxmodel, sigsynfluxmodel, sigfluxmodel  =   np.std(modelbumparr)*1.0,  np.std(modelbbodyarr)*1.0, np.std(modelsynarr)*1.0, np.std(modelfluxarr)*1.0
    return blackbodyfluxmodel,  twocomp_BB_params, synmodelwbb, bumpmodelwbb, blackbodymodel, synfluxmodel, bbbumpfluxmodel, mychisquarewbb, mmodel, sigbbbumpfluxmodel, sigblackbodyfluxmodel, sigsynfluxmodel, sigfluxmodel
