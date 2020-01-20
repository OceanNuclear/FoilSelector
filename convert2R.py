from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from matplotlib import pyplot as plt
import pandas as pd
import json
from openmc.data import NATURAL_ABUNDANCE
import uncertainties
from ReadData import main_read
from collapx import flux_conversion, main_collapse, MeV, read_apriori_and_gs_df

UNWANTED_RADITAION = ['alpha', 'n', 'sf', 'p']
HPGE_ERROR = False
IRRADIATION_DURATION = 180 #seconds
CM2_BARNS_CONVERSION = 1E-24 # convert flux to barns
COUNT_TIME = 180 # seconds
USE_NATURAL_ABUNDANCE = True
VERBOSE = False

# apriori_per_eV has unit cm^-2 s^-1 eV^-1
def simple_decay_correct(half_life, transit_time): #accounts for the decay between the end of the irradiation period up to the recording period.
    # assume time to get to the detector = 60s
    decay_correct_factor = 1
    decay_correct_factor = 2.0**(-transit_time/half_life)
    return decay_correct_factor #should actually return a uncertainties.core.Variable

def fispact_decay_correct():# accounts for the decay etc. during the non-zero irradiation time.
    '''
    Number of atoms at the end of the irraidation period
    when modelled as a non-zero irradiation time/ when modelled as a flash irradiation.
    '''
    return uncertainties.core.Variable(1,0) #again, preferably returns an uncertainties.core.Variable

def total_decay_correct(product_list, transit_time=60):
    decay_correct_factor = simple_decay_correct(product_list[0].half_life, transit_time)*fispact_decay_correct() # take only the first item in the product list.
    #use the first version of the file encoutnered to extract the half_life and its uncertainties.
    return decay_correct_factor

def geometric_efficiency():
    return 0.5

def efficiency_curve(energy_in_eV):
    assert type(energy_in_eV)==uncertainties.core.Variable
    intrinsic_efficiency = 1
    return intrinsic_efficiency

def width(energy_in_eV):
    assert type(energy_in_eV)==uncertainties.core.Variable
    FWHM_in_eV = 1
    return FWHM_in_eV

def integrate_count(decay_constant, count_time=COUNT_TIME):
    return count_time/( 1-exp(-decay_constant*count_time) )

def add_unc(c):
    return uncertainties.core.Variable(c, sqrt(c))

def compute_weighted_count_rate(count_rates_with_unc):
    for count_rate in count_rates_with_unc:
        assert type(count_rate)==uncertainties.core.Variable
    if len(count_rates_with_unc)>0:
        new_variance = 1/ sum([ (1/count.s**2) for count in count_rates_with_unc])
        unnormalized_mean = sum([ (count.n/(count.s**2)) for count in count_rates_with_unc ])
        best_fit_N = uncertainties.core.Variable(unnormalized_mean* new_variance, sqrt(new_variance))
        return best_fit_N
    else:
        return None

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

def R_conversion_main(reaction_and_radiation, apriori_and_unc, R_file, rr_file, spectra_file):
    spectra_json = {}
    _counted_rr, _R_matrix, r_name_list= [], [], [] # the first tow lists are un-sorted, therefore I prefer to leave them as private variables.

    for rname, rnr_file in reaction_and_radiation.items():
        if USE_NATURAL_ABUNDANCE:
            try:
                N_target = number_of_atoms_of_element() * NATURAL_ABUNDANCE[rname.split('-')[0]]
            except KeyError:
                N_target = 0
                if VERBOSE:
                    print(f"{rname}'s parent is not naturally occurring, and is therefore excluded")
        else:
            N_target = number_of_atoms_of_element()
        if N_target !=0: #filtering out the non-naturally occuring elements
            all_peaks = []
            for dec_files in rnr_file.decay:# use whichever version that has a more comprehensive (longer) list of gamma radiation
                try:
                    len_rad = [len(d.spectra['gamma']['discrete']) for d in dec_files]
                    d = np.argmax(len_rad)
                    dec = dec_files[d]
                except KeyError:
                    dec = dec_files[0]
                all_peaks_of_this_prod = []
                # for name in ranked_names:
                #     reaction_and_radiation[name].decay[product_num][decay_file_version]
                for rad, specific_spec in dec.spectra.items():
                    if rad in UNWANTED_RADITAION:
                        print(f"Warning: unwanted radiation={rad} is found for {rname}. See {spectra_file} for its intensity")
                    if rad=='gamma' or rad=='xray':
                        if specific_spec['continuous_flag']!='discrete':
                            print(f"!! THIS REACTION {rname} HAS A CONTINUOUS ENERGY FLAG FOR ITS GAMMA/XRAY!")
                        for peak in specific_spec['discrete']:
                            all_peaks_of_this_prod.append(peak)
                all_peaks.append(all_peaks_of_this_prod)
                spectra_json[rname] = dec.spectra
            #deterministic part:
            # calculate the expected number of nuclides left at the beginning of the measurement
            N_infty = rnr_file.sigma @ apriori_and_unc['value'].values * IRRADIATION_DURATION * N_target *CM2_BARNS_CONVERSION# unit: number of atoms # assuming flash irradiation
            #   start to split-up the prediction by products, i.e. if there are multiple products, then we'll have to sum them.
            N_0, decay_correct_factors_with_unc, decay_constant = [], [], []
            for product in rnr_file.decay:
                decay_correct_factors_with_unc.append( total_decay_correct(product) ) # correct for flash irradiation assumption, and then for decay during transit.
                decay_constant.append( product[0].decay_constant ) #choose the first version of that product that comes up.
                N_0.append( N_infty * decay_correct_factors_with_unc[-1].n )# unit: number of atoms
                
            if HPGE_ERROR: #(Only go into this for loop if we're propagating errors starting from the gamma radiations.')
                #create the following lists once for different product within the same reaction.
                for j in range(len(all_peaks)):
                    Omega, intrinsic_efficiency, intensities, decay_constant, count_rates, counts = [], [], [], [], [], [] # len=len(all_peaks_of_this_prod)=i
                    counts_with_unc, count_rates_with_unc, Nk_with_unc = [], [], [] # len=len(all_peaks_of_this_prod)=i
                    N0_with_unc_of_each_prod = [] #len=len(all_peaks)=j
                    for i in range(len(all_peaks[j])):
                        peak = all_peaks[j][i]
                        # Sum over the count rate from each peak.
                        Omega.append(               geometric_efficiency() )            # dimensionless
                        intrinsic_efficiency.append(efficiency_curve(peak['energy']) )  # dimensionless
                        intensities.append(         peak['intensity'] )                 # dimensionless
                        count_rates.append(         N_0[j] * intensities[i].n * intrinsic_efficiency[i].n * Omega[i] ) # number of atoms s^-1
                        counts.append(              count_rates[i] * integrate_count(decay_constant[j], COUNT_TIME).n ) #number of atoms
                        
                        #end of deterministic part; start propagating errors
                        # Now pretend I've finished acquiring these numbers from the HPGe detector, and obtained the associated uncertainties.
                        counts_with_unc.append(     add_unc(counts) )
                        count_rates_with_unc.append(counts_with_unc[i]/ integrate_count(decay_constant[j], COUNT_TIME) )
                        #theoretically the integrate_count factor should covaries with the half_life used below; but we're simplifying things here and assuming that they are two independent variables right now.
                        Nk_with_unc.append(         count_rates[i]/(intensities[i] * intrinsic_efficiency[i] * Omega) )
                    N0_with_unc_of_each_prod.append( compute_weighted_count_rate(Nk_with_unc) )

            else: #assume 100% detection efficiency
                error_on_N_0 = sqrt(N_0)*(1/sqrt(COUNT_TIME))
                N0_with_unc_of_each_prod = ary([uncertainties.core.Variable(N_0[i], error_on_N_0[i]) for i in range(len(N_0))])
            
            all_products = [productlist[0] for productlist in rnr_file.decay]

            #add the nuclide into the list
            if len(N0_with_unc_of_each_prod)>0 and all([type(i)!=type(None) for i in N0_with_unc_of_each_prod]): #only add it if all of its products produces detectible radiation.
                N_infty_with_unc = sum((N0_with_unc_of_each_prod/decay_correct_factors_with_unc))/len(N0_with_unc_of_each_prod) # assume identical branching ratios
                # N_infty_with_unc = (N0_with_unc_of_each_prod/decay_correct_factors_with_unc)[0] # assume the majority of it becomes the first product listed
                # N_infty_with_unc = (N0_with_unc_of_each_prod/decay_correct_factors_with_unc)[-1]# assume the majority of it becomes the last product listed
                
                _counted_rr.append([N_infty_with_unc.n, N_infty_with_unc.s])
                print(rname, "->", "|".join([prod.nuclide['name'] for prod in all_products])," is added")
                _R_matrix.append( CM2_BARNS_CONVERSION * ary(rnr_file.sigma) )
                r_name_list.append(rname)
            else:
                print(f"Not every product of {rname} has a detectible radiation. Therefore it is excluded.")

            if len(all_products)>1: # take only the first item in each product list as the representative nuclide information dictionary.
                print(rname, "has more than 1 product, namely", [prod.nuclide['name'] for prod in all_products] )
                print("Assuming there is the same probability of decaying into each of these products,")
                print("Parent:", rnr_file.parent)
                print("Daughters are as follows:")
                for nuc in [prod.nuclide for prod in all_products]:
                    print(nuc)
                
    R = pd.DataFrame(_R_matrix, index=r_name_list)
    rr = pd.DataFrame(_counted_rr, index=r_name_list, columns=['value','uncertainty'])
    # new_rname_order = ary(r_name_list)[np.argsort( _counted_rr)][::-1]
    # R = R.loc[new_rname_order]
    # rr=rr.loc[new_rname_order]

    with open(spectra_file, mode='w', encoding='utf-8') as f:
        json.dump(turn_Var_into_str(spectra_json), f)
    print(f"The spectra for each reaction is saved at {spectra_file}")
    
    R.to_csv(R_file)
    rr.to_csv(rr_file)
    primer_sentence = "Number of nuclides transmuted after irradiation of "+str(number_of_atoms_of_element())+" of that "
    if USE_NATURAL_ABUNDANCE:
        primer_sentence+= "element in its natural composition"
    else:
        primer_sentence+= "isotope"
    print(primer_sentence+"by the un-normalized apriori spectrum for {0}s, along with its measurement uncertainties, is saved in {1}".format(str(IRRADIATION_DURATION), rr_file))
    print(len(rr), "reactions are considered in total.")
    print(R_file, "contains the response matrix with the unit: cm^2")
    print("I.e. The number of transmutation per parent nuclide per cm^-2 s^-1 of neutron flux, integrated over 1 second = R")

    return R, rr, spectra_json

if __name__=='__main__':

    rdict, dec_r, all_mts = main_read()
    apriori_and_unc, gs_array = read_apriori_and_gs_df('output/gs.csv', 'output/apriori.csv', 'integrated', apriori_multiplier=1E5, gs_multipliers=MeV) # apriori is read in as eV^-1
    reaction_and_radiation = main_collapse(apriori_and_unc, gs_array, rdict, dec_r)
    R, rr, spectra_json = R_conversion_main(reaction_and_radiation, apriori_and_unc, "output/Scaled_R_matrx.csv", "output/rr.csv", "output/spectra.json")

    # Tasks
    '''
    3. Then implement gamma overlap checks, and allow it to be imported into the next file.
    4. Check that they're all of the right shape
    '''
    # Tests:
    '''
    1. The HPGE_ERROR=True should perform slightly worse (with larger uncertainties, but same means) than HPGE_ERROR=False
    2. Increasing COUNT_TIME should asymptotically improve the uncertainties to a limit.
    3. Fast decaying should be affected more by increasing IRRADIATION_DURATION than slow decaying.
    4. Use the integration function of the Tabulated1D properly.
    '''