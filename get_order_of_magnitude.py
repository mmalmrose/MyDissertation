'''takes an input array that will be plotted and gets the order of magnitude
  of the median value of the array.  For example np.array([1.4e+13,  1.7e+14, 1.1e+15])
  will return 10^14'''
def get_order_of_magnitude(input_array):
    import numpy as np
    orderofmag = (10**np.fix(np.log10(np.nanmean(input_array))))
    return orderofmag
