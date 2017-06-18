import numpy as np
def twocomponentjustbump(bumpnormal, bumpindex):
    bigbluebump = np.float64(bumpnormal*nu**(bumpindex))
    return bigbluebump
