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
MeV = 1E6

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

def area_between_2_pts(xy1,xy2, xi,scheme):
    x1, y1, x2, y2, xi = np.hstack([xy1, xy2, xi]).flatten()
    assert x1<=xi<=x2, "xi must be between x1 and x2"
    x_ = xi-x1
    if x1==x2:
        return 0.0  #catch all cases with zero-size width bins
    if y1==y2 or scheme==1:# histogramic/ flat interpolation
        return y1*x_ #will cause problems in all log(y) interpolation schemes if not caught
    
    dy, dx = y2-y1, x2-x1
    logy, logx = [bool(int(i)) for i in bin(scheme-2)[2:].zfill(2)]
    if logx:
        assert(all(ary([x1,x2,xi])>0)), "Must use non-zero x values for these interpolation schemes"
        lnx1, lnx2, lnxi = ln(x1), ln(x2), ln(xi)
        dlnx = lnx2 - lnx1
    if logy:
        assert(all(ary([y1,y2])>0)), "Must use non-zero y values for these interpolation schemes"
        lny1, lny2 = ln(y1), ln(y2)
        dlny = lny2 - lny1

    if scheme==2: # linx, liny
        m = dy/dx
        if xi==x2:
            return dy*dx/2 + y1*dx
        else:
            return y1*x_ + m*x_**2 /2
    elif scheme==3: # logx, liny
        m = dy/dlnx
        if xi==x2:
            return y1*dx + m*(x2*dlnx-dx)
        else:
            return y1*x_ + m*(xi*(lnxi-lnx1)- x_)
            return (y1 - m*lnx1)*x_ + m*(-x_+xi*lnxi-x1*lnx1)
    elif scheme==4: # linx, logy
        m = dlny/dx
        if xi==x2:
            return 1/m *dy
        else:
            return 1/m *y1*(exp(x_*m)-1)
    elif scheme==5:
        m = dlny/dlnx
        if m==-1:
            return y1 * x1 * (lnxi-lnx1)
        if xi==x2:
            return y1/(m+1) * ( x2 * (x2/x1)**m - x1 )
        else:
            return y1/(m+1) * ( xi * (1 + x_/x1)**m - x1 )
    else:
        raise AssertionError("a wrong interpolation scheme {0} is provided".format(scheme))

class Integrate():
    def __init__(self, sigma):
        self.x = sigma.x
        self.y = sigma.y
        self.xy = ary([self.x, self.y])
        assert all(np.diff(self.x)>=0), "the x axis must be inputted in ascending order"
        self.interpolations = []
        self.next_area = []
        for i in range(len(self.x)-1):
            break_region = np.searchsorted(sigma.breakpoints, i+1, 'right')
            self.interpolations.append(sigma.interpolation[break_region])
            # region_num = np.searchsorted(i, self.breakpoints, 'right')
            # scheme = self.interpolation[region_num]
            x1, x2 = sigma.x[i:i+2]
            xy1 = ary([x1, sigma.y[i]])
            xy2 = ary([x2, sigma.y[i+1]])
            self.next_area.append(area_between_2_pts(xy1, xy2, x2, scheme=self.interpolations[i] ))
    def __call__(self, a, b):
        assert np.shape(a)==np.shape(b), "There must be as many starting points as the ending points to the integrations."
        if isinstance(a, Iterable):
            return ary([self(ai, bi) for (ai,bi) in zip(a,b)]) #catch all iterables cases.
            # BTW, don't ever input a scalar into __call__ for an openmc.data.Tabulated1D function.
            # It's going to return a different result than if you inputted an Iterable.
            # Integrate(), on the other hand, doesn't have this bug.
        
        a, b = np.clip([a,b], min(self.x), max(self.x)) # assume if no data is recorded at the low energy limit.
        ida, idb = self.get_region(a), self.get_region(b)
        total_area = self.next_area[ida:idb]
        total_area.append(-area_between_2_pts(self.xy[:,ida], self.xy[:,ida+1], a, self.interpolations[ida]))
        total_area.append(area_between_2_pts(self.xy[:,idb], self.xy[:,idb+1], b, self.interpolations[idb]))
        return fsum(total_area)

    def get_region(self, x):
        idx = np.searchsorted(self.x, x, 'right')-1
        assert all([x<=max(self.x)]),"Out of bounds of recorded x! (too high)"
        # assert all([0<=idx]), "Out of bounds of recorded x! (too low)" #doesn't matter, simply assume zero. See __call__
        idx -= x==max(self.x) * 1 # if it equals eactly the upper limit, we need to minus one in order to stop it from overflowing.
        return np.clip(idx, 0, None)

def flux_conversion(flux_in, gs_in_eV, in_fmt, out_fmt):
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
        assert in_fmt=="per eV", "the input format 'i' must be one of the following 4='integrated'|'PUL'(per unit lethargy)|'per (M)eV'"
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

def easy_logspace(start, stop, **kwargs):
    logstart, logstop = np.log10([start, stop])
    return np.logspace(logstart, logstop, **kwargs)

def integrate_continuous_flux(continuous_func, gs):
    inte = Integrate(continuous_func)
    return [inte(*minmax) for minmax in gs]

def get_integrated_apriori_value_only(directory):
    full_file_path = join(directory, "integrated_apriori.csv")
    integrated_apriori = pd.read_csv(full_file_path, sep="±|,", engine='python')
    return integrated_apriori['value'].values

def get_continuous_flux(directory):
    full_file_path = join(directory, "continuous_apriori.pkl")
    with open(full_file_path,'rb') as f:
        apriori_func = pickle.load(f)
    return apriori_func

def get_gs_ary(directory):
    full_file_path = join(directory, "gs.csv")
    gs = pd.read_csv(full_file_path)
    return gs.values

def list_dir_csv(directory):
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

def ask_yn_question(question):
    while True:
        answer = input(question+"('y','n')")
        if answer.lower() in ['yes', 'y']:
            return True
        elif answer.lower() in ['no', 'n']:
            return False
        else:
            print(f"Option {answer} not recognized; please retry:")

def scale_to_eV_interactive(gs_ary):
    gs_ary_fmt_unit_question = f"Were the group structure values \n{gs_ary}\n given in 'eV', 'keV', or 'MeV'?"
    #scale the group structure up
    gs_ary_fmt_unit = ask_question(gs_ary_fmt_unit_question, ['eV', 'keV', 'MeV'])
    if gs_ary_fmt_unit=='MeV':
        gs_ary = gs_ary*M
    elif gs_ary_fmt_unit=='keV':
        gs_ary = gs_ary*k
    return gs_ary

def convert_arbitrary_gs_from_means(gs_means):
    first_bin_size, last_bin_size = np.diff(gs_means)[[0,-1]]
    mid_points = gs_means[:-1]+np.diff(gs_means)/2
    gs_min = np.hstack([gs_means[0]-first_bin_size/2, mid_points])
    gs_max = np.hstack([mid_points, gs_means[-1]+last_bin_size/2])    
    return ary([gs_min, gs_max])

def ask_for_gs(directory):
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
            gs_min, gs_max = convert_arbitrary_gs_from_means(gs_mean)
    elif gs_fmt == 'class boundaries':
        gs_min = get_column_interactive(directory, 'lower bounds of the energy groups')
        gs_max = get_column_interactive(directory, 'upper bounds of the energy groups')
    gs_ary = scale_to_eV_interactive( ary([gs_min, gs_max]).T )
    return gs_ary

minmax = lambda lst: (min(lst),max(lst))

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
    apriori_copy = apriori.copy() # leave a copy to be used in section 1.5
    fmt_question = "What format was the a priori provided in?('per eV', 'per keV', 'per MeV', 'PUL', 'integrated')"
    in_unit = ask_question(fmt_question, ['PUL','per MeV', 'per keV', 'per eV', 'integrated'])
    
    print("1.2 Conversion into continuous format---------------------------------------------------------------------------------------")
    group_or_point_question = "Is this a priori spectrum given in 'group-wise' (fluxes inside discrete bins) or 'point-wise' (continuous function requiring interpolation) format?"
    group_or_point = ask_question(group_or_point_question, ['group-wise','point-wise'])
    
    print("1.3 Reading the energy values associated with the a priori------------------------------------------------------------------")
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

    plt.plot(x:=np.linspace(*minmax(E_values)), continuous_apriori(x))
    plt.show()

    print("1.4 [optional] Modifying the a priori---------------------------------------------------------------------------------------")
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
                plt.plot(x:=np.linspace(*minmax(E_values)), continuous_apriori(x))
                plt.show()
                can_stop = ask_yn_question("Is this satisfactory?")
                if can_stop:
                    print("Retrying...")
                    break
            except ValueError as e:
                print(e)

    #increase the flux up to a set total flux
    total_flux = Integrate(continuous_apriori)(min(E_values), max(E_values))
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
        total_flux = Integrate(continuous_apriori)(min(E_values), max(E_values))
        plt.plot(x:=np.linspace(*minmax(E_values)), continuous_apriori(x))
        plt.show()
    print(f"{total_flux = }")

    print("1.5 [optional] adding an uncertainty to the a priori-------------------------------------------------------------------------")
    error_present = ask_yn_question("Does the a priori spectrum comes with an associated error (y-error bars) on itself?")
    if not error_present:
        pass
    if error_present:
        error_series = get_column_interactive(sys.argv[-1], 'error (which should be of the same shape as the a priori spectrum input)')
        fractional_error = error_series/apriori_copy
        fractional_error[np.isnan(fractional_error)] = 0
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

    print("2. Load in group structure--------------------------------------------------------------------------------------------------")
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
                gs_bounds = easy_logspace(E_min, E_max, num=E_num+1)
            elif E_interp=='lin-space':
                gs_bounds = np.linspace(E_min, E_max, num=E_num+1)
            gs_min, gs_max = gs_bounds[:-1], gs_bounds[1:]
            gs_array = ary([gs_min, gs_max]).T
        elif gs_source=='from file':
            gs_array = ask_for_gs(sys.argv[-1])

    # plot the histogramic version of it once
    integrated_flux = integrate_continuous_flux(continuous_apriori, gs_array)
    if error_present:
        uncertainty = (integrate_continuous_flux(continuous_apriori_upper,gs_array) - integrate_continuous_flux(continuous_apriori_lower,gs_array))/2
        print("An (inaccurate) estimate of the error is provided as well. This is likely overestimated (by a factor of ~ sqrt(2)) as it does not obey the rules of error propagation properly.")
    
    gs_df = pd.DataFrame(gs_array, columns=['min', 'max'])
    gs_df.to_csv(join(sys.argv[-1], 'gs.csv'), index=False)

    if not error_present:
        apriori_vector_df = pd.DataFrame(integrated_flux, columns=['value'])
    elif error_present:
        apriori_vector_df = pd.DataFrame(integrated_flux, columns=['value', 'uncertainty'])
    
    apriori_vector_df.to_csv(join(sys.argv[-1], 'integrated_apriori.csv'), index=False)
    
    with open('continuous_apriori.pkl', 'wb') as f:
        pickle.dump(continuous_apriori, f)