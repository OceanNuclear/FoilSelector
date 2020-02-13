from numpy import exp, cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from matplotlib import pyplot as plt
import pandas as pd
import os, sys, glob
join, base = os.path.join, os.path.basename
import csv
from openmc.data import INTERPOLATION_SCHEME, Tabulated1D
from math import fsum
import pickle
from typing import Iterable
k, M = 1E3, 1E6
from convert_flux import *
'''
An improvement upon convert_flux.py
'''

def list_dir(directory):
    fnames = glob.glob(join(directory, "*.csv"))
    print("########################")
    print("------------------------")
    for f in fnames:
        print(base(f))
    print("------------------------")
    print("########################")
    return fnames

def open_csv(fname):
    sniffer = csv.Sniffer()
    if sniffer.has_header(fname):
        df = pd.read_csv(fname, sep=",|±", skipinitialspace=True, engine='python')
    else:
        df = pd.read_csv(fname, sep=",|±", header=None, skipinitialspace=True, engine='python')
    return df, df.columns

def get_column_interactive(directory, datatypename, first_time_use=False):
    while True:
        try:
            prompt = ""
            if not first_time_use:
                prompt = "(Can be the same file as above)"
            fname = input(f"Which of the above file contains values for {datatypename}?"+prompt)
            df, col = open_csv(join(directory,fname))
            print("Opened\n", df.head(), "\n...")
            colname = input(f"Please input the index/name of the column where {datatypename} is/are contained.\n(column name options include {list(col)})")
            if colname in col:
                col_i = colname
                break
            else:
                col_i = col[int(colname)]
                break
        except Exception as e:
            print(e, "perhaps the file name/ column name is wrong/ index is too high. Please try again.")
    dataseries = df[col_i]
    print(f"using\n{dataseries.head()}\n...")
    return dataseries.values

def ask_question(question, expected_answer_list, check=True):
    while True:
        answer = input(question)
        if answer in expected_answer_list:
            break
        else:
            if not check:
                break
            print(f"Option {answer} not recognized; please retry:")
    print()
    return answer

def scale_to_eV_interactive(gs_ary):
    gs_ary_fmt_unit_question = f"Were the group structure values \n{gs_ary}\n given in 'eV', 'keV', or 'MeV'?"
    #scale the group structure up
    gs_ary_fmt_unit = ask_question(gs_ary_fmt_unit_question, ['eV', 'keV', 'MeV'])
    if gs_ary_fmt_unit=='MeV':
        gs_ary = gs_ary*M
    elif gs_ary_fmt_unit=='keV':
        gs_ary = gs_ary*k
    return gs_ary

# def extend_pair(lst1, lst2):
#     '''for extending the last bin's value when plotting/interpolating '''
#     return np.hstack([lst1,lst1[-1]]), np.hstack([lst2,lst2[-1]])

def convert_arbitrary_gs_from_means(gs_means):
    first_bin_size, last_bin_size = np.diff(gs_means)[[0,-1]]
    mid_points = gs_means[:-1]+np.diff(gs_means)/2
    gs_min = np.hstack([gs_means[0]-first_bin_size/2, mid_points])
    gs_max = np.hstack([mid_points, gs_means[-1]+last_bin_size/2])    
    return ary([gs_min, gs_max])

minmax = lambda lst: (min(lst),max(lst))

if __name__=='__main__':
    print("The following inputs are needed")
    print("1. The a priori spectrum")
    print("2. The energy value given along with the a priori spectrum")
    print("3. The group structure to be used (can be the same as 2.)")
    print()
    print("The relevant data will be retrieved from the following csv files.")
    print(f"The following csv files are found in {sys.argv[-1]}:")
    list_dir(sys.argv[-1])
    
    apriori = get_column_interactive(sys.argv[-1], "a priori spectrum's values (ignoring the uncertainty)", first_time_use=True)
    fmt_question = "What format was the a priori provided in?('per eV', 'per keV', 'per MeV', 'PUL', 'integrated')"
    in_unit = ask_question(fmt_question, ['PUL','per MeV', 'per keV', 'per eV', 'integrated'])
    
    group_or_point_question = "Is this apriori spectrum given in 'group-wise' (fluxes inside discrete bins) or 'point-wise' (continuous function requiring interpolation) format?"
    group_or_point = ask_question(group_or_point_question, ['group-wise','point-wise'])
    

    if group_or_point=='group-wise':
        apriori_gs_fmt_question = "Were the flux values provided along with the mean energy of the bin ('class mark'), or the upper and lower ('class boundaries')?"
        apriori_gs_fmt = ask_question(apriori_gs_fmt_question, ['class mark', 'class boundaries'])
        if apriori_gs_fmt == 'class mark':
            apriori_gs_mean = get_column_interactive(sys.argv[-1], "a priori spectrum's mean energy of each bin")
            bin_sizes = np.diff(apriori_gs_mean)
            assert all(bin_sizes>0), "The apriori spectrum must be given in ascending order of energy."
            #deal with two special cases: lin space and log space
            if all(np.isclose(np.diff(bin_sizes), 0, atol=1E-3)): #second derivative = 0
                print("equal spacing in energy space detected")
                average_step = np.mean(bin_sizes)
                start_point, end_point = apriori_gs_mean[[0,-1]]
                apriori_gs_min = np.linspace(start_point-average_step/2, end_point-average_step/2, num=len(apriori_gs_mean), endpoint=True)
                apriori_gs_max = np.linspace(start_point+average_step/2, end_point+average_step/2, num=len(apriori_gs_mean), endpoint=True)
            elif all(np.isclose(np.diff(np.diff(apriori_gs_mean)), 0, atol=1E-3)): #second derivative of log(E) = 0
                print("equal spacing in log(E) space detected")
                average_step = np.mean(np.diff(np.log10(apriori_gs_mean)))
                start_point, end_point = np.log10(apriori_gs_mean[[0,-1]])
                apriori_gs_min = np.logspace(start_point-average_step/2, end_point-average_step/2, num=len(apriori_gs_mean), endpoint=True)
                apriori_gs_max = np.logspace(start_point+average_step/2, end_point+average_step/2, num=len(apriori_gs_mean), endpoint=True)
            else: #neither log-spaced nor lin-spaced
                apriori_gs_min, apriori_gs_max = convert_arbitrary_gs_from_means(apriori_gs_mean)
        elif apriori_gs_fmt=='class boundaries':
            apriori_gs_min = get_column_interactive(sys.argv[-1], 'lower bounds of the energy groups')
            apriori_gs_max = get_column_interactive(sys.argv[-1], 'upper bounds of the energy groups')
        apriori_gs = ary([apriori_gs_min, apriori_gs_max]).T
        apriori_gs = scale_to_eV_interactive(apriori_gs)

        apriori = flux_conversion(apriori, apriori_gs, in_unit, 'per eV')

        E_values = np.hstack([apriori_gs[:,0], apriori_gs[-1,1]])
        apriori = np.hstack([apriori, apriori[-1]])
        scheme = val_to_key_lookup(INTERPOLATION_SCHEME, 'histogram')
    elif group_or_point=='point-wise':
        E_values = get_column_interactive(sys.argv[-1], 'energy associated with each data point of the a priori')
        E_values = scale_to_eV_interactive(E_values)
        #convert to per eV format
        if in_unit=='PUL':
            apriori = apriori * E_values
        elif in_unit=='per MeV':
            apriori = apriori * M
        elif in_unit=='per keV':
            apriori = apriori * k
        elif in_unit=='per eV':
            pass # nothing needs to change
        elif in_unit=='integrated':
            raise NotImplementedError("Continuous data should never be an integrated flux!")

        print("The scheme available for interpolating between data points are\n", INTERPOLATION_SCHEME, "\ne.g. linear-log denotes linear in y, logarithmic in x")
        scheme = int(ask_question("What scheme should be used to interpolate between the two points? (type the index)", [str(i) for i in INTERPOLATION_SCHEME.keys()]))
        
        if scheme==1:
            apriori = np.hstack([apriori, apriori[-1]])
            E_values = np.hstack([E_values, E_values[-1]+np.diff(E_values)[-1]])
        apriori_gs = ary([E_values[:-1], E_values[1:]]).T
    
    continuous_apriori = Tabulated1D(E_values, apriori, breakpoints=[len(apriori),], interpolation=[scheme,])

    plt.plot(x:=np.linspace(*minmax(E_values)), continuous_apriori(x))

    # scale the peak up and down (while keeping the total flux the same)
    flux_shifting_question = "Would you like to shift the flux up or down?('y','n')"
    to_shift_or_not_to_shift = ask_question(flux_shifting_question, ['y','n'])
    if to_shift_or_not_to_shift=='y':
        print("new energy scale = (scale_factor) * old energy scale + offset")
        while True:
            try:
                scale_factor=float(input("scale_factor"))
                offset = float(input("offset"))
                E_values = scale_factor * E_values + offset
                break
            except ValueError as e:
                print(e)
    elif to_shift_or_not_to_shift=='n':
        pass
    continuous_apriori = Tabulated1D(E_values, apriori, breakpoints=[len(apriori),], interpolation=[scheme,])

    #increase the flux up to a set total flux
    total_flux = Integrate(continuous_apriori)(min(E_values), max(E_values))
    flux_scaling_question = f"{total_flux = }, would you like to scale it up/down?('y','n')"
    to_scale_or_not_to_scale = ask_question(flux_scaling_question,['y','n'])

    if to_scale_or_not_to_scale=='y':
        while True:
            try:
                new_total_flux = float(input("Please input the new total flux:"))
                break
            except ValueError as e:
                print(e)
        apriori = apriori * new_total_flux/total_flux
        continuous_apriori = Tabulated1D(E_values, apriori, breakpoints=[len(apriori)], interpolation=[scheme,])
        total_flux = Integrate(continuous_apriori)(min(E_values), max(E_values))
    elif to_scale_or_not_to_scale=='n':
        pass
    print(f"{total_flux = }")

    plt.plot(x:=np.linspace(*minmax(E_values)), continuous_apriori(x))

    same_gs_question = "Should a different group structure than the apriori_gs (entered above) be used?('y','n')"
    same_gs = ask_question(same_gs_question, ['y', 'n'])
    
    if same_gs=='y':
        gs_array = apriori_gs
    elif same_gs=='n':
        gs_source_question = "Would you like to read the gs_bounds 'from file' or manually create an 'evenly spaced' group structure?"
        gs_source = ask_question(gs_source_question, ['from file', 'evenly spaced'])
        if gs_source=='evenly spaced':
            print("Using the naïve approach of dividing the energy/lethargy axis into equally spaced bins.")
            print("For reference, the current minimum and maximum of the apriori spectrum are", *minmax(continuous_apriori.x))
            while True:
                try:
                    E_min = float(ask_question("What is the desired minimum energy for the group structure used?", [], check=False))
                    E_max = float(ask_question("What is the desired minimum energy for the group structure used?", [], check=False))
                    spacing_prompt = "Would you like to perform a 'log-space'(equal spacing in energy space) or 'lin-space'(equal spacing in lethargy space) interpolation between these two limits?"
                    E_interp = ask_question(spacing_prompt, ['log-space', 'lin-space'])
                    E_num = int(ask_question("How many bins would you like to have?", [], check=False))
                    break
                except ValueError as e:
                    print(e)
            if E_interp=='log-space':
                gs_bounds = easy_logspace(E_min, E_max, num=E_num+1)
            elif E_interp=='lin-space':
                gs_bounds = np.linspace(E_min, E_max, num=E_num+1)
            gs_min, gs_max = gs_bounds[:-1], gs_bounds[1:]
        elif gs_source=='from file':
            gs_min = get_column_interactive('lower bounds of the groups', first_time_use=True)
            gs_max = get_column_interactive('upper bounds of the groups')
        
        gs_array = ary([gs_min, gs_max]).T

    # plot the histogramic version of it once
    integrate_continuous_flux(continuous_apriori, gs_array)
    #save gs
    #save continuous_apriori
    #save integrated_flux