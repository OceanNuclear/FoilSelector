from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from matplotlib import pyplot as plt
from convert2R import *
import json

def num_crossover( func, start, stop, resolution=400, depth=0, max_depth=4):
    range_to_search = np.linspace(start, stop, num=resolution, endpoint=True)
    arg_vec = np.argwhere(np.diff([ func(i) for i in range_to_search ]))
    if len(arg_vec)>1:
        return True # yes, there is more than one crossing points
    elif len(arg_vec)==1:
        if depth<max_depth:
            return num_crossover(func, range_to_search[crossing], range_to_search[crossing+1], resolution, depth+1, max_depth)
        else:
            return False #No, even after max_depth number of zooms, there is still only one crossing point.
    else:
        raise ValueError("No crossing point found!")

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

def get_peak_distances(sorted_energies):
    num_peaks= len(sorted_energies)
    dist_array = np.repeat(sorted_energies, num_peaks).reshape([num_peaks,-1])
    dist_array -= sorted_energies
    return np.triu(dist_array.T, 1)

def get_confusable_peaks(spec_file):
    adj_matrix = get_peak_distances()
    return

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