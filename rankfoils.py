from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from matplotlib import pyplot as plt
from convert2R import *

new_rname_order = ary(r_name_list)[np.argsort( _counted_rr)][::-1]

R = R.loc[new_rname_order]
rr=rr.loc[new_rname_order]