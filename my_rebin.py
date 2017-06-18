def my_rebin( a, newshape ):
#Rebin an array to a new shape.
    import os 
    import numpy as np
    import scipy as sp
    from scipy import misc
    from StringIO import StringIO
    from astropy.io import fits as pyfits
    #print newshape
    assert len(a) == newshape
    slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
    coordinates = mgrid[slices]
    indices = coordinates.astype('i')   #choose the biggest smaller integer index
    return a[tuple(indices)]
