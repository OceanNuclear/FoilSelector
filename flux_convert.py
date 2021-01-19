# native python functions
import csv
from math import fsum
import os, sys, glob
join, base = os.path.join, os.path.basename
# numpy functions
import numpy as np
from numpy import exp
from numpy import log as ln
from numpy import array as ary
# plotting and pandas
from matplotlib import pyplot as plt
import pandas as pd
# openmc
from openmc.data import INTERPOLATION_SCHEME, Tabulated1D
# custom functions
from misc_library import Integrate, MeV, detabulate, tabulate
k, M = 1E3, 1E6

'''
An improvement upon convert_flux.py
'''

def val_to_key_lookup(dic, val):
    for k,v in dic.items():
        if v==val:
            return k
def get_scheme(scheme):
    return val_to_key_lookup(INTERPOLATION_SCHEME, scheme)
loglog = get_scheme('log-log')
histogramic = get_scheme('histogram')

def flux_conversion(flux_in, gs_in_eV, in_fmt, out_fmt):
    """
    Convert flux from a one representation into another.

    Parameters
    ----------
    flux_in : flux to be converted into out_fmt. A pd.DataSeries/DataFrame, a numpy array or a list of (n) flux values.
              The meaning of each of the n flux values is should be specified by the parameter in_fmt (see in_fmt below).

    gs_in_eV : group-structure with shape = (n, 2), which denotes the upper and lower energy boundaries of the bin

    in_fmt, out_fmt : string describing the format of the flux when inputted/outputted
                      Accepted argument for fmt's:
                        "per MeV",
                        "per eV",
                        "per keV",
                        "integrated",
                        "PUL"
    """
    if isinstance(flux_in, pd.DataFrame) or isinstance(flux_in, pd.Series): # check type
        flux= flux_in.values.T
    else:
        flux=flux_in
    #convert all of them to per eV
    if in_fmt=='per MeV':
        flux_per_eV = flux/MeV
    elif in_fmt=='integrated':
        flux_per_eV = flux/np.diff(gs_in_eV, axis=1).flatten()
    elif in_fmt=='PUL':
        leth_space = np.diff(ln(gs_in_eV), axis=1).flatten()
        flux_integrated = flux*leth_space
        flux_per_eV = flux_conversion(flux_integrated, gs_in_eV, 'integrated', 'per eV')
    elif in_fmt=='per keV':
        flux_per_eV = flux/1E3
    else:
        assert in_fmt=="per eV", "the input format 'i' must be one of the following 4='integrated'|'PUL'(per unit lethargy)|'per (k/M)eV'"
        flux_per_eV = flux
    # convert from per eV back into 
    if out_fmt=='per MeV':
        flux_out= flux_per_eV*MeV
    elif out_fmt=='integrated':
        flux_out= flux_per_eV * np.diff(gs_in_eV, axis=1).flatten()
    elif out_fmt=='PUL':
        leth_space = np.diff(ln(gs_in_eV), axis=1).flatten()
        flux_integrated = flux_conversion(flux_per_eV, gs_in_eV,'per eV', 'integrated')
        flux_out= flux_integrated/leth_space
    else:
        assert out_fmt=='per eV', "the input format 'i' must be one of the following 4='integrated'|'PUL'(per unit lethargy)|'per (M)eV'"
        flux_out= flux_per_eV
    
    #give it back as the original type
    if isinstance(flux_in, pd.DataFrame) or isinstance(flux_in, pd.Series):
        flux_out = type(flux_in)(flux_out)
        name_or_col = "column" if isinstance(flux_in, pd.DataFrame) else "name"
        setattr(flux_out, name_or_col, getattr(flux_in, name_or_col))
    return flux_out

def intuitive_logspace(start, stop, *args, **kwargs):
    """
    plug in the actual start and stop limit to the logspace function, without having to take log first.
    """
    logstart, logstop = np.log10([start, stop])
    return np.logspace(logstart, logstop, *args, **kwargs)

def get_integrated_apriori_value_only(directory):
    """
    Find integrated_apriori.csv in a specified directory and load in its values.
    Written and saved in this module so that it can be imported by other modules in the future.
    (i.e. written here for future compatibility)
    """
    full_file_path = join(directory, "integrated_apriori.csv")
    integrated_apriori = pd.read_csv(full_file_path, sep="±|,", engine='python', skipinitialspace=True)
    return integrated_apriori['value'].values

def get_continuous_flux(directory):
    """
    retrieve the continuous flux distribution (openmc.data.Tabulated1D object, stored in "continuous_apriori.csv") from a directory,
    and turn it back into an openmc.data.Tabulated1D object.
    """
    detabulated_apriori_df = pd.read_csv(join(directory, "continuous_apriori.csv"))
    detabulated_apriori = { "x":detabulated_apriori_df["x"].tolist(),
                            "y":detabulated_apriori_df["y"].tolist(),
                            "interpolation":detabulated_apriori_df["interpolation"].tolist()}
    detabulated_apriori["interpolation"].pop()
    return tabulate(detabulated_apriori)

def get_gs_ary(directory):
    """
    Find the group structure of a specified directory and load in its values.
    Written and saved in this module so that it can be iported by other modules in the future.
    (I.e. written here for future compatibility).
    """
    full_file_path = join(directory, "gs.csv")
    gs = pd.read_csv(full_file_path, skipinitialspace=True)
    return gs.values

def list_dir_csv(directory):
    """
    Pretty print the list of all csv in a specified directory.
    """
    fnames = glob.glob(join(directory, "*.csv"))
    print("########################")
    print("------------------------")
    for f in fnames:
        print(base(f))
    print("------------------------")
    print("########################")
    return fnames

def open_csv(fname):
    """
    General function that opens any csv, where ",|±" are all intepreted as separators, and Header is optional.
    """
    sniffer = csv.Sniffer()
    if sniffer.has_header(fname):
        df = pd.read_csv(fname, sep=",|±", skipinitialspace=True, engine='python')
    else:
        df = pd.read_csv(fname, sep=",|±", header=None, skipinitialspace=True, engine='python')
    return df, df.columns

def get_column_interactive(directory, datatypename, first_time_use=False):
    """
    Ask the user for the column in a csv file within the specified {directory}, containing the {datatypename}.
    Keep asking until it is successfully found.
    Parameters
    ----------
    directory : location to look for csv.
    datatypename : name of the datatype which is displayed to the user when asking the question.
    first_time_use : if False, modifies the prompt question by appending the string "(Can be the same file as above)",
                     so that the user intuitively understands that the same file as the one used to answer the question in the previous call to this function can be used.
    """
    while True:
        try:
            prompt = ""
            if not first_time_use:
                prompt = "(Can be the same file as above)"
            fname = input(f"Which of the above file contains values for the {datatypename}?"+prompt)
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
    """
    Ask the user a multiple choice question.
    Parameters
    ----------
    question : entire string of the question.
    expected_answer_list : list of strings which are the expected answers.
                           If the user gives an answer that isn't included the list, and check=True,
                           then their answer will be discarded and the quesiton will be asked again until a matching answer is found.
    check : see expected_answer_list
    """
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

def ask_yn_question(question):
    """
    Ask a yes no question
    parameters
    ----------
    question : string containing the question to be displayed.

    question displayed
    ------------------
    question+"('y'/'n')"

    accepted inputs
    ---------------
    answer.lowercase() must be "yes", "y", "no" or "n".
    e.g.
    yes: ["yes", "y", "Yes", "YES", "Y"]
    no: ["no", "n", "No", "NO", "N"]

    returns
    -------
    Boolean (True/False)
    """
    while True:
        answer = input(question+"('y','n')")
        if answer.lower() in ['yes', 'y']:
            return True
        elif answer.lower() in ['no', 'n']:
            return False
        else:
            print(f"Option {answer} not recognized; please retry:")

def percentile_of(minmax_of_range, percentile=50):
    """
    Find the value that's a certain percent above the minimum of that range.
    e.g. percentile_of([-1,1], 10) = -1 + (2)*0.1 = -0.8.

    parameters
    ----------
    minmax_of_range: list containing the minimum value and maximum value of the range.
    percentile : what percent above the minium is wanted.
    """
    perc = float(percentile/100)
    return min(minmax_of_range) + perc*abs(np.diff(minmax_of_range))

def scale_to_eV_interactive(gs_ary):
    """
    Scale the group structure values so that it describes the group structure in the correct unit (eV).

    returns
    -------
    gs_ary : group structure array, of the same shape as the input gs_ary, but now each value describes the energy bin bounds in eV.
    """
    gs_ary_fmt_unit_question = f"Were the group structure values \n{gs_ary}\n given in 'eV', 'keV', or 'MeV'?"
    #scale the group structure up
    gs_ary_fmt_unit = ask_question(gs_ary_fmt_unit_question, ['eV', 'keV', 'MeV'])
    if gs_ary_fmt_unit=='MeV':
        gs_ary = gs_ary*M
    elif gs_ary_fmt_unit=='keV':
        gs_ary = gs_ary*k
    return gs_ary

def convert_arbitrary_gs_from_means(gs_means):
    """
    Create a group structure (n bins, with upper and lower bounds each) from a list of n numbers.
    This is done by taking the first (n-1) numbers as the upper bounds of the last (n-1) bins,
    and the last (n-1) numbers as the lower bounds of the first (n-1) bins.
    The first bin's lower bound and the last bin's upper bound is obtained by
    extrapolating the bin width of the second bin and the penultimate bin respectively.

    parameters
    ----------
    gs_means : a list of n numbers, describing the class-mark of each bin.
    """
    first_bin_size, last_bin_size = np.diff(gs_means)[[0,-1]]
    mid_points = gs_means[:-1]+np.diff(gs_means)/2
    gs_min = np.hstack([gs_means[0]-first_bin_size/2, mid_points])
    gs_max = np.hstack([mid_points, gs_means[-1]+last_bin_size/2])    
    return ary([gs_min, gs_max]).T

def ask_for_gs(directory):
    """
    Check directory for a file containing a group structure; and interact with the program user to obtain said group structure.
    """
    gs_fmt_question = "Were the flux values provided along with the mean energy of the bin ('class mark'), or the upper and lower ('class boundaries')?"
    gs_fmt = ask_question(gs_fmt_question, ['class mark', 'class boundaries'])
    if gs_fmt == 'class mark':
        gs_mean = get_column_interactive(directory, "a priori spectrum's mean energy of each bin")
        bin_sizes = np.diff(gs_mean)
        assert all(bin_sizes>0), "The a priori spectrum must be given in ascending order of energy."
        #deal with two special cases: lin space and log space
        if all(np.isclose(np.diff(bin_sizes), 0, atol=1E-3)): #second derivative = 0
            print("equal spacing in energy space detected")
            average_step = np.mean(bin_sizes)
            start_point, end_point = gs_mean[[0,-1]]
            gs_min = np.linspace(start_point-average_step/2, end_point-average_step/2, num=len(gs_mean), endpoint=True)
            gs_max = np.linspace(start_point+average_step/2, end_point+average_step/2, num=len(gs_mean), endpoint=True)
        elif all(np.isclose(np.diff(np.diff(gs_mean)), 0, atol=1E-3)): #second derivative of log(E) = 0
            average_step = np.mean(np.diff(np.log10(gs_mean)))
            start_point, end_point = np.log10(gs_mean[[0,-1]])
            gs_min = np.logspace(start_point-average_step/2, end_point-average_step/2, num=len(gs_mean), endpoint=True)
            gs_max = np.logspace(start_point+average_step/2, end_point+average_step/2, num=len(gs_mean), endpoint=True)
        else: #neither log-spaced nor lin-spaced
            gs_min, gs_max = convert_arbitrary_gs_from_means(gs_mean).T
    elif gs_fmt == 'class boundaries':
        while True:
            gs_min = get_column_interactive(directory, 'lower bounds of the energy groups')#, first_time_use=True)
            gs_max = get_column_interactive(directory, 'upper bounds of the energy groups')
            if (gs_min<gs_max).all():
                break
            else:
                print("The left side of the bins must have strictly lower energy than the right side of the bins. Please try again:")
    gs_ary = scale_to_eV_interactive( ary([gs_min, gs_max]).T )
    return gs_ary

if __name__=='__main__':
    print("The following inputs are needed")
    print("1. The a priori spectrum, and the energies at which those measurements are taken.")
    print("2. The group structure to be used in the investigation that follows.")
    print()
    print("The relevant data will be retrieved from the following csv files.")
    print(f"The following csv files are found in {sys.argv[-1]}:")
    list_dir_csv(sys.argv[-1])
    
    print("1.1 Reading the a priori values---------------------------------------------------------------------------------------------")
    apriori = get_column_interactive(sys.argv[-1], "a priori spectrum's values (ignoring the uncertainty)", first_time_use=True)
    apriori_copy = apriori.copy() # leave a copy to be used in section 4.
    fmt_question = "What format was the a priori provided in?('per eV', 'per keV', 'per MeV', 'PUL', 'integrated')"
    in_unit = ask_question(fmt_question, ['PUL','per MeV', 'per keV', 'per eV', 'integrated'])
    
    print("1.2 Conversion into continuous format---------------------------------------------------------------------------------------")
    group_or_point_question = "Is this a priori spectrum given in 'group-wise' (fluxes inside discrete bins) or 'point-wise' (continuous function requiring interpolation) format?"
    group_or_point = ask_question(group_or_point_question, ['group-wise','point-wise'])
    
    print("2. Reading the energy values associated with the a priori------------------------------------------------------------------")
    if group_or_point=='group-wise':
        apriori_gs = ask_for_gs(sys.argv[-1])
        apriori = flux_conversion(apriori, apriori_gs, in_unit, 'per eV')

        E_values = np.hstack([apriori_gs[:,0], apriori_gs[-1,1]])
        scheme = histogramic

    elif group_or_point=='point-wise':
        E_values = get_column_interactive(sys.argv[-1], 'energy associated with each data point of the a priori')
        E_values = scale_to_eV_interactive(E_values)
        #convert to per eV format
        if in_unit=='PUL':
            apriori = apriori * E_values
        elif in_unit=='per MeV':
            apriori = apriori / M
        elif in_unit=='per keV':
            apriori = apriori / k
        elif in_unit=='per eV':
            pass # nothing needs to change
        elif in_unit=='integrated':
            raise NotImplementedError("Continuous data should never be an integrated flux!")

        print("The scheme available for interpolating between data points are\n", INTERPOLATION_SCHEME, "\ne.g. linear-log denotes linear in y, logarithmic in x")
        scheme = int(ask_question("What scheme should be used to interpolate between the two points? (type the index)", [str(i) for i in INTERPOLATION_SCHEME.keys()]))
        
        if scheme==histogramic:
            E_values = np.hstack([E_values, E_values[-1]+np.diff(E_values)[-1]])
        apriori_gs = ary([E_values[:-1], E_values[1:]]).T

    if scheme==histogramic:
        apriori = np.hstack([apriori, apriori[-1]])
    continuous_apriori = Tabulated1D(E_values, apriori, breakpoints=[len(apriori),], interpolation=[scheme,])

    # plot in per eV scale
    plt.plot(x:=np.linspace(min(E_values), max(E_values), num=200), continuous_apriori(x))
    plt.show()
    
    #plot in lethargy scale
    ap_plot = flux_conversion(apriori[:-1], apriori_gs, 'per eV', 'PUL')
    plt.step(E_values, np.hstack([ap_plot, ap_plot[-1]]), where='post')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('lethargy scale plot')
    plt.show()

    print("3. [optional] Modifying the a priori---------------------------------------------------------------------------------------")
    # scale the peak up and down (while keeping the total flux the same)
    flux_shifting_question = "Would you like to shift the energy scale up/down?"
    to_shift = ask_yn_question(flux_shifting_question)
    if not to_shift:
        pass
    elif to_shift:
        print("new energy scale = (scale_factor) * old energy scale + offset")
        while True:
            try:
                scale_factor=float(input("scale_factor="))
                offset = float(input("offset="))
                E_values = scale_factor * E_values + offset
                continuous_apriori = Tabulated1D(E_values, apriori, breakpoints=[len(apriori),], interpolation=[scheme,])
                plt.plot(x:=np.linspace(*minmax(E_values), num=200), continuous_apriori(x))
                plt.show()
                can_stop = ask_yn_question("Is this satisfactory?")
                if can_stop:
                    break
                print("Retrying...")
            except ValueError as e:
                print(e)

    #increase the flux up to a set total flux
    total_flux = Integrate(continuous_apriori).definite_integral(min(E_values), max(E_values))
    flux_scaling_question = f"{total_flux = }, would you like to scale it up/down?"
    to_scale = ask_yn_question(flux_scaling_question)

    if not to_scale:
        pass
    elif to_scale:
        while True:
            try:
                new_total_flux = float(input("Please input the new total flux:"))
                break
            except ValueError as e:
                print(e)
        apriori = apriori * new_total_flux/total_flux
        continuous_apriori = Tabulated1D(E_values, apriori, breakpoints=[len(apriori)], interpolation=[scheme,])
        total_flux = Integrate(continuous_apriori).definite_integral(min(E_values), max(E_values))
        plt.plot(x, continuous_apriori(x))
        plt.show()
    print(f"{total_flux = }")

    print("4. [optional] adding an uncertainty to the a priori-------------------------------------------------------------------------")
    error_present = ask_yn_question("Does the a priori spectrum comes with an associated error (y-error bars) on itself?")
    if error_present:
        error_series = get_column_interactive(sys.argv[-1], "error (which should be of the same shape as the a priori spectrum input)")
        # allow the error to be inputted in either fractional error or absolute error.
        absolute_or_fractional = ask_question("does this describe the 'fractional' or 'absolute' error?", ['fractional', 'absolute'])
        if absolute_or_fractional=='fractional':
            fractional_error = error_series
        else:
            fractional_error = np.nan_to_num(error_series/apriori_copy)
        if scheme==histogramic:
            fractional_error_ext = np.hstack([fractional_error, fractional_error[-1]])
            error = fractional_error_ext * apriori
        else:
            error = fractional_error * apriori
        continuous_apriori_lower = Tabulated1D(E_values, apriori-error, breakpoints=[len(apriori)], interpolation=[scheme,])
        continuous_apriori_upper = Tabulated1D(E_values, apriori+error, breakpoints=[len(apriori)], interpolation=[scheme,])
        plt.fill_between(x, continuous_apriori_upper(x), continuous_apriori_lower(x))
        plt.plot(x, continuous_apriori(x), color='orange')
        plt.show()

    print("5. Load in group structure--------------------------------------------------------------------------------------------------")
    diff_gs_question = "Should a different group structure than the apriori_gs (entered above) be used?"
    diff_gs = ask_yn_question(diff_gs_question)
    
    if not diff_gs: # same gs as the a priori
        gs_array = apriori_gs
    elif diff_gs:
        gs_source_question = "Would you like to read the gs_bounds 'from file' or manually create an 'evenly spaced' group structure?"
        gs_source = ask_question(gs_source_question, ['from file', 'evenly spaced'])
        if gs_source=='evenly spaced':
            print("Using the naïve approach of dividing the energy/lethargy axis into equally spaced bins.")
            print("For reference, the current minimum and maximum of the a priori spectrum are", *minmax(continuous_apriori.x))
            while True:
                try:
                    E_min = float(ask_question("What is the desired minimum energy for the group structure used?", [], check=False))
                    E_max = float(ask_question("What is the desired maximum energy for the group structure used?", [], check=False))
                    spacing_prompt = "Would you like to perform a 'log-space'(equal spacing in energy space) or 'lin-space'(equal spacing in lethargy space) interpolation between these two limits?"
                    E_interp = ask_question(spacing_prompt, ['log-space', 'lin-space'])
                    E_num = int(ask_question("How many bins would you like to have?", [], check=False))
                    break
                except ValueError as e:
                    print(e)
            if E_interp=='log-space':
                gs_bounds = intuitive_logspace(E_min, E_max, num=E_num+1)
            elif E_interp=='lin-space':
                gs_bounds = np.linspace(E_min, E_max, num=E_num+1)
            gs_min, gs_max = gs_bounds[:-1], gs_bounds[1:]
            gs_array = ary([gs_min, gs_max]).T
        elif gs_source=='from file':
            gs_array = ask_for_gs(sys.argv[-1])
    
    fig, ax = plt.subplots()
    ax.set_ylabel("E(eV)")
    ax.set_xlabel("flux(per eV)")
    ax.plot(x, continuous_apriori(x))
    ybounds = ax.get_ybound()
    for limits in gs_array:
        ax.errorbar(x=np.mean(limits), y=percentile_of(ybounds, 10), xerr=np.diff(limits[::-1])/2, capsize=30, color='black')
        # Draw the group structure at ~10% of the height of the graph.
    plt.show()

    # plot the histogramic version of it once
    integrated_flux = Integrate(continuous_apriori).definite_integral(*gs_array.T)
    if error_present:
        uncertainty = (Integrate(continuous_apriori_upper).definite_integral(*gs_array.T) -
                        Integrate(continuous_apriori_lower).definite_integral(*gs_array.T) )/2
        print("An (inaccurate) estimate of the error is provided as well. If a different group structure than the input file's group structure is used, then this error likely overestimated (by a factor of ~ sqrt(2)) as it does not obey the rules of error propagation properly.")
    
    gs_df = pd.DataFrame(gs_array, columns=['min', 'max'])
    gs_df.to_csv(join(sys.argv[-1], 'gs.csv'), index=False)

    if not error_present:
        apriori_vector_df = pd.DataFrame(integrated_flux, columns=['value'])
    elif error_present:
        apriori_vector_df = pd.DataFrame(ary([integrated_flux, uncertainty]).T, columns=['value', 'uncertainty'])
    
    apriori_vector_df.to_csv(join(sys.argv[-1], 'integrated_apriori.csv'), index=False)
    
    # save the continuous a priori distribution (an openmc.data.Tabulated1D object) as a csv, by specifying the interpolation scheme as well.
    detabulated_apriori = detabulate(continuous_apriori)
    detabulated_apriori["interpolation"].append(0) # 0 is a placeholder, it doesn't correspond to any interpolation scheme, but is an integer so that pandas wouldn't treat it differently; unlike using None, which would force the entire column to become floats.
    detabulated_apriori_df = pd.DataFrame(detabulated_apriori)
    detabulated_apriori_df["interpolation"] = detabulated_apriori_df["interpolation"].astype(int)
    detabulated_apriori_df.to_csv(join( sys.argv[-1], "continuous_apriori.csv"), index=False) # x already acts pretty well as the index.

    print("Preprocessing completed. The outputs are saved to:")
    print("group structure => gs.csv,")
    print("flux vector => integrated_apriori.csv, ")
    print("continuous apriori (an openmc.data.Tabulated1D object) => continuous_apriori.csv")