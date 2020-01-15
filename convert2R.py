from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from matplotlib import pyplot as plt
from collapx import *
import uncertainties
from openmc.data import NATURAL_ABUNDANCE
import json
UNWANTED_RADITAION = ['alpha', 'n', 'sf', 'p']
HPGE_ERROR = False
IRRADIATION_DURATION = 180 #seconds
BARNS_CM2_CONVERSION = 1E-24
#apriori_per_eV has unit cm^-2 s^-1 eV^-1
def simple_decay_correct(half_life, transit_time):
    # assume time to get to the detector = 60s
    decay_correct_factor = 1
    decay_correct_factor = 2.0**(-transit_time/half_life)
    return decay_correct_factor #should actually return a uncertainties.core.Variable

def fispact_decay_correct():
    return 1 #again, preferably returns an uncertainties.core.Variable

def total_decay_correct(product_list, transit_time=60):
    decay_correct_factor = simple_decay_correct(product_list[0].half_life, transit_time)*fispact_decay_correct()
    #use the first version of the file encoutnered to extract the half_life and its uncertainties.
    return decay_correct_factor

def count_rate(decay_constant):
    return 1

def efficiency_curve(energy_in_eV):
    assert type(energy_in_eV)==uncertainties.core.Variable
    intrinsic_efficiency = 1
    return intrinsic_efficiency

def width(energy_in_eV):
    assert type(energy_in_eV)==uncertainties.core.Variable
    FWHM_in_eV = 1
    return FWHM_in_eV

def geometric_efficiency():
    return 0.5

def give_count_rate_with_uncertainties( c, recording_time=None):
    return uncertainties.core.Variable(c, sqrt(c))

def compute_weighted_count_rate(list_of_count_rates_with_errors):
    for count_rate in list_of_count_rates_with_errors:
        assert type(count_rate)==uncertainties.core.Variable
    best_fit_N = [sum([])/sum([]) for something in list_of_count_rates_with_errors]
    return ary(best_fit_N)#I don't know how yet.

def number_of_atoms_of_element():
    return 6.02214076E23 # currently set to one mole

def turn_Var_into_str(data): #operable on dictionaries.
    if type(data)==uncertainties.core.Variable or type(data)==uncertainties.core.AffineScalarFunc:
        data = str(data.n) + str("+/-")+str(data.s)
    elif type(data)==dict:
        for k,v in data.items():
            data[k] = turn_Var_into_str(v)
    elif type(data)==list: #then it is a class
        data = [turn_Var_into_str(i) for i in data]
    else:
        pass
    return data

spectra_file = "output/spectrum.json"
print(f"The spectra for each reaction converts is saved at {spectra_file}")
R_file = "output/Scaled_R_matrx.csv"
print(f"The scaled response matrix {R_file} converts count flux to count rates.")
rr_file = "output/rr.csv"
print(f"The reaction rates are saved at {rr_file}")

_spectra_json  = {}
_counted_rr, _R_matrix, r_name_list= [], [], []

for rname, rnr_file in reaction_and_radiation.items():
    try:
        N_target = number_of_atoms_of_element() * NATURAL_ABUNDANCE[rname.split('-')[0]]
        for dec in np.concatenate(rnr_file.decay): #WON'T WORK WHEN THERE ARE MULTIPLE PRODUCTS! Have to correct this later.
            all_peaks = []
            # for name in ranked_names:
            #     reaction_and_radiation[name].decay[product_num][decay_file_version]
            for rad, specific_spec in dec.spectra.items():
                if rad in UNWANTED_RADITAION:
                    print(f"Warning: unwanted radiation={rad} is found for {rname}. See {spectra_file} for its intensity")
                if rad=='gamma' or rad=='xray':
                    if specific_spec['continuous_flag']!='discrete':
                        print(f"!! THIS REACTION {rname} HAS A CONTINUOUS ENERGY FLAG FOR ITS GAMMA/XRAY!")
                    for peak in specific_spec['discrete']:
                        all_peaks.append(peak)
        #deterministic part:
        # calculate the expected number of nuclides left at the beginning of the measurement
        N_infty = rnr_file.sigma @ apriori_and_unc['value'].values * IRRADIATION_DURATION * N_target *BARNS_CM2_CONVERSION# unit: number of atoms
        #   start to split-up the prediction by products, i.e. if there are multiple products, then we'll have to sum them.
        decay_correct_factor_with_unc = ary([ total_decay_correct(product) for product in rnr_file.decay])
        decay_correct_factor = ary([ total_decay_correct(product).n for product in rnr_file.decay])
        N_0 = N_infty * decay_correct_factor# unit: number of atoms
        # Right now I'm duplicating N_0 by however many supposed reaction products it has.
        if HPGE_ERROR:
            # calculate the count rate for each peak
            Omega = geometric_efficiency() # dimensionless
            intrinsic_efficiency = [ efficiency_curve(peak.energy) for peak in all_peaks ] # dimensionless
            branching_ratio = [ peak.intensity for peak in all_peaks ] # dimensionless
            decay_constant = [ dec.decay_constant for peak in all_peaks ] # s^-1
            count_rates = [ N_0 * branching_ratio[i].n * intrinsic_efficiency[i].n*Omega for i in range(len(all_peaks)) ] # number of atoms s^-1
            #end of deterministic part; start propagating errors
            
            # Now pretend I've finished acquiring these numbers from the HPGe detector:
            count_rates = [ give_count_rate_with_uncertainties(c) for c in count_rates ] #start by creating uncertainties
            #    then use _EACH_ peak to calculate what was the N_0
            N_k = [ count_rates[i]/(branching_ratio[i]*intrinsic_efficiency[i]*Omega) for i in range(len(all_peaks)) ]
            N_0_with_unc = compute_weighted_count_rate(N_k)

        else:
            error_on_N_0 = (N_0 - N_0)*(1/sqrt(200))
            N_0_with_unc = ary([uncertainties.core.Variable(N_0[i], error_on_N_0[i]) for i in range(len(N_0))])
            N_infty_with_unc = sum(N_0_with_unc/decay_correct_factor_with_unc)
        _counted_rr.append(N_infty_with_unc)
        print(rname," is added")
        _R_matrix.append( N_infty * ary(rnr_file.sigma))
        r_name_list.append(rname)
        _spectra_json[rname] = [ turn_Var_into_str(dec.spectra) for dec in np.concatenate(rnr_file.decay) ]
    except KeyError:
        pass


R = pd.DataFrame(_R_matrix, index=r_name_list)
rr = pd.DataFrame(_counted_rr, index=r_name_list)

new_rname_order = ary(r_name_list)[np.argsort( _counted_rr)][::-1]
R = R.loc[new_rname_order]
rr=rr.loc[new_rname_order]

with open(spectra_file, mode='w', encoding='utf-8') as f:
    json.dump(_spectra_json, f)
R.to_csv(R_file)
rr.to_csv(rr_file)