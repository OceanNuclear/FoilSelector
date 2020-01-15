from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from matplotlib import pyplot as plt
from convert2R import *
'''
    Achieved:
-Filter fissile/fissionable materials
-collapsed the cross sections into the proper group structures, and returned them as list of lists.
-get the microscopic cross-sections
-Account for transit
-(partially) translate it into reaction rates rr
-Assign uncertainty on rr
-print warning if there are alpha etc.

    To be finished:

-Select foil properly: use determinant of the covariance matrix.
-Show that the confusion matrix is the identity matrix for over-determined unfolding.
-Select foils in the case of over-determined unfolding.
-allow user to input error on the apriori

-*Account for detector characteristics as well
-Get the covariance matrices, plus other radionuclides produced in MF=10 (using FISPACT)
-Account for decay over irradiation period (using FISPACT)

    Lower priority:
-Add warning messages about melting point etc.
-Add option about something 
-un-spaghettify code
'''