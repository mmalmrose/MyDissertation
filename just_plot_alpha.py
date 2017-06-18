#!/usr/bin/python
import numpy as np
myfile = '../outputfiles/1222+216/VARBBB/textout/alphaoutput.csv'
mystuff= np.genfromtxt(myfile, dtype=None)
date = mystuff.item()[0]
alpha= mystuff.item()[1]
sigalpha= mystuff.item()[2]
fluxalpha = mystuff.item()[3]
sigfluxalpha= mystuff.item()[4]
print myalpha
