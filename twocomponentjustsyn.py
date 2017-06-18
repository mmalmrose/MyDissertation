import numpy as np
def twocomponentjustsyn(nu, synnormal, alpha):
    synchrotron = np.float64(synnormal*nu**(alpha*(-1.0)))
    return synchrotron
