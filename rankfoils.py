from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from matplotlib import pyplot as plt
from convert2R import THRESHOLD_ENERGY, load_R_rr_radiation
import pandas as pd
import json
from scipy.optimize import curve_fit
from os.path import join, basename
import sys

def FWHM_ftting_func(E, a, b):
    return a + sqrt(b*E)

def get_FWHM_file(directory):
    E, FWHM_val, FWHM_error = pd.read_csv(join(directory, "FWHM.csv"), comment='#').values.T
    return E, FWHM_val, FWHM_error

def get_func(E, FWHM_val, **kwargs):
    [a0,b0], cov = curve_fit(FWHM_ftting_func, E, FWHM_val, **kwargs)
    func = lambda E: FWHM_ftting_func(E, a0, b0)
    setattr(func, 'a', a0)
    setattr(func, 'b', b0)
    return func

def get_peak_sigma(FWHM_func, E):
    return FWHM_func(E)/2.355

# def num_crossover( func, start, stop, resolution=400, depth=0, max_depth=4):
#     range_to_search = np.linspace(start, stop, num=resolution, endpoint=True)
#     arg_vec = np.argwhere(np.diff([ func(i) for i in range_to_search ]))
#     if len(arg_vec)>1:
#         return True # yes, there is more than one crossing points
#     elif len(arg_vec)==1:
#         if depth<max_depth:
#             return num_crossover(func, range_to_search[crossing], range_to_search[crossing+1], resolution, depth+1, max_depth)
#         else:
#             return False #No, even after max_depth number of zooms, there is still only one crossing point.
#     else:
#         raise ValueError("No crossing point found!")

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

def get_confusable_peaks(E1, E2):
    # if within 3 FWHM of each other:
    return

def min_thickness():
    detectible_limit = 3*noise_level
    return

def perform_fit(directory):
    E, FWHM_val, FWHM_error = get_FWHM_file(directory)
    Esmooth = np.linspace(*THRESHOLD_ENERGY)
    fit = get_func(E, FWHM_val, sigma=FWHM_error)
    print(f"{fit.a=}, {fit.b=}")
    plt.errorbar(E, FWHM_val, yerr=FWHM_error, linestyle='', capsize=5)
    plt.plot(Esmooth, fit(Esmooth))
    plt.ylim(0)
    plt.xlabel('energy (eV)')
    plt.ylabel('FWHM (eV)')
    plt.title("Fit of the FWHM of the peaks")
    plt.show()
    return fit

def ask_yn_question(question):
    while True:
        answer = input(question)
        if answer.lower() in ['y', 'yes']:
            return True
        elif answer.lower() in ['n', 'no']:
            return False
        else:
            print(f"Sorry {answer} not understood")

def append_into_dict(d, key, element):
    if key in d.keys():
        d[key].append(element)
    else:
        d[key]=[element,]

if __name__=="__main__":
    while True:
        FWHM_func = perform_fit(sys.argv[-1])
        if ask_yn_question("Satisfied with the fit?"):
            break
    print("Checking the spectrum for overlaps...")
    R, rr, spectra_json = load_R_rr_radiation(sys.argv[-1])
    spectrum = {rname:spectra_json[rname] for rname in rr.index}
    all_rad,all_prod = {}, {}
    for rname, reaction in spectrum.items():
        for daughter, decay in reaction.items():
            append_into_dict(all_prod, rname, daughter)
            for radname, radiation in decay.items():
                if radname in ['xray', 'gamma']:
                    for peak in radiation['discrete']:
                        append_into_dict(all_rad, rname)
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