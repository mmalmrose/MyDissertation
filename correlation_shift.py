#!/usr/bin/python
import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2, fftshift
import random
def cross_correlation_using_fft(x, y):
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)
 
# shift &lt; 0 means that y starts 'shift' time steps before x # shift &gt; 0 means that y starts 'shift' time steps after x
def compute_shift(x, y):
    assert len(x) == len(y)
    c = cross_correlation_using_fft(x, y)
    assert len(c) == len(x)
    zero_index = int(len(x) / 2) - 1
    shift = zero_index - np.argmax(c)
    return shift
    

for n in range(1000, 1050, 7):
    for s in range(-5, 5):        
        a = [random.random() for _ in xrange(n)] # big random sequence of values
        b = a
        if s >= 1:
            a = a[s:]
            b = b[:-s]
        elif s <= -1:            
            a = a[:s]
            b = b[-s:]
        #assert s_optimal == s
       
myshift = compute_shift(a, b)
#print a, b
print 'myshift', myshift
#print 's', s_optimal
