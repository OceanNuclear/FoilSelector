from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from matplotlib import pyplot as plt
from convert2R import *
import json

AREA = pi*3.0**2 # The bigger the area, the less thickness is required to reach a detectible limit, thus self-shielding distortion of the true spectrum.
#The only disadvantage to having a foil with too much area is that it must be very thin,
#perhaps impossible-to-manufature-ly thin, not overexpose the detector.

def get_histogram(spec_file):
    with open(spec_file, 'r') as f:
        spec = json.load(f)
    rad_list = []
    for r, prods in spec.items(): 
        for prod, rads in prods.items(): 
            for rad, info in rads.items(): 
                if rad=='gamma' or rad=='xray': 
                    for peak in info['discrete']: 
                        rad_list.append(uncertainties.core.Variable(*peak['energy'].split('+/-'))) 
    return rad_list

def check_overlapping_peaks( energies, intensities, warning_name):
    return

def total_count_rate(radiation_dict):
    for rad, specific_rad in radiation_dict.items():
        if rad=='gamma' or rad=='xray':
            if specific_rad['energy']>THRESHOLD:
                pass

def min_thickness():
    detectible_limit = 3*noise_level
    return

def max_thickness(deadtime_percent=5):
    saturation_limit
    return 
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