from numpy import exp, cos, arccos, sin, arctan, tan, pi, sqrt; import numpy as np; tau = 2*pi
from numpy import array as ary
from numpy import log as ln
from matplotlib import pyplot as plt
import pandas as pd
import json
from openmc.data import NATURAL_ABUNDANCE
import uncertainties
from collapx import flux_conversion, MeV, read_rnr, ReactionAndRadiation
from flux_convert import get_integrated_apriori_value_only
import os, sys, glob
import copy
from openmc.data import Tabulated1D

UNWANTED_RADITAION = ['alpha', 'n', 'sf', 'p']
SIMULATE_GAMMA_DETECTOR = True
MAX_OUT_PHOTOPEAK_EFF = False
IRRADIATION_DURATION = 1 #seconds # Try all different ones
TRANSIT_TIME = 10*60 # time it takes to turn off the beam, take it out, and put it on the detector.
COUNT_TIME = 1*3600 # seconds
USE_NATURAL_ABUNDANCE = True
THRESHOLD_ENERGY = [100E3, 4.6E6] # eV # energy range of gamma that can be detected # taking the highest L1 absorption edge as the lower limit
MIN_COUNT_PER_MOLE_PARENT = 1E3 # Ignore peaks with less than this count.
EVALUATE_THERMAL = True
MIN_CURVE_CONTR_MULTIPLER = 1 # allows the minimum thickness to be more lenient when unfolding.
# MIN_COUNT_RATE_PER_MOLE_PARENT = 0 # Ignore peaks with less than this count rate.
# A more accurate program would take into account Compton plateau background from higher energy peaks of the same daughter; but I can't be asked to program in the Compton part of the detector response too.
# Also, if a program can do that, it can probably also DECONVOLUTE the entire spectrum so that the Comptoms are also counted, therefore achieving a higher efficiency anyways.
CM2_BARNS_CONVERSION = 1E-24 # convert flux to barns

VERBOSE = False
FOCUS_MODE = False

# apriori_per_eV has unit cm^-2 s^-1 eV^-1
def simple_decay_correct(half_life, transit_time): #accounts for the decay between the end of the irradiation period up to the recording period.
    # assume time to get to the detector = 60s
    decay_correct_factor = 1
    decay_correct_factor = 2.0**(-transit_time/half_life)
    return decay_correct_factor #should actually return a uncertainties.core.Variable

def fispact_decay_correct(decay_constant, flat_profile=True):# accounts for the decay etc. during the non-zero irradiation time.
    '''
    Number of atoms at the end of the irraidation period
    when modelled as a non-zero irradiation time/ when modelled as a flash irradiation.
    '''
    #Batesman equation
    # return uncertainties.core.Variable(1,0) #again, preferably returns an uncertainties.core.Variable
    if type(decay_constant) in [uncertainties.core.Variable, uncertainties.core.AffineScalarFunc]:
        decay_constant = decay_constant.n
    simple_corr_fac = (1-exp(-decay_constant*IRRADIATION_DURATION))/(decay_constant*IRRADIATION_DURATION)
    # The uncertainty part is not implemented yet
    return simple_corr_fac

def total_decay_correct(product_versions, transit_time):
    decay_correct_factor = simple_decay_correct(product_versions[0].half_life, transit_time)*fispact_decay_correct(product_versions[0].decay_constant) # take only the first item in the product list.
    #use the first version of the file encoutnered to extract the half_life and its uncertainties.
    return decay_correct_factor

def geometric_efficiency():
    return 0.5

def HPGe_efficiency_curve_generator(working_dir, deg=4, cov=True):
    '''
    according to Knoll (equation 12.32), polynomial fit in log-log space is the best.
    This program is trying to do the same.
    This polynomial in log-log space fit is confirmed by
    @article{kis1998comparison,
    title={Comparison of efficiency functions for Ge gamma-ray detectors in a wide energy range},
    author={Kis, Zs and Fazekas, B and {\"O}st{\"o}r, J and R{\'e}vay, Zs and Belgya, T and Moln{\'a}r, GL and Koltay, L},
    journal={Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment},
    volume={418},
    number={2-3},
    pages={374--386},
    year={1998},
    publisher={Elsevier}
    }
    '''
    file_location = glob.glob(os.path.join(working_dir, '*photopeak_efficiency*.csv'))[0]
    datapoints = pd.read_csv(file_location)
    assert "energ" in datapoints.columns[0].lower(), "The file must contain a header. Energy (eV/MeV) has to be placed in the first column"
    E, eff = datapoints.values.T[:2]
    if 'MeV' in datapoints.columns[0] or 'MeV' in file_location:
        E = MeV*E
    if datapoints.values.T[2:].shape[0]>0: # assume that column is the error
        sigma = datapoints.values.T[2]
        print("Using the 3rd column of ", file_location, r" as error on the efficiency measurements (\sigma)")
        cov = 'unscaled'
        w = 1/sigma**2
    else:
        w = None
    from scipy import polyfit
    p = polyfit(ln(E), ln(eff), deg, w=w, cov=cov) #make sure that the curve points downwards at high energy 
    if cov:
        p, pcov = p
    #iteratively try increasing degree of polynomial fits.
    #Currently the termination condition is via making sure that the curve points downwards at higher energy (leading coefficient is negative).
    #However, I intend to change the termination condition to Bayesian Information Criterion instead.
    if not p[0]<0:
        mindeg, maxdeg = 2,6
        for i in range(mindeg, maxdeg+1):
            p = polyfit(ln(E), ln(eff), deg, w=w, cov=cov)            
            if cov:
                p, pcov = p
            if p[0]<0:
                print("a {0} order polynomial fit to the log(energy)-log(efficiency) curve is used instead of the default {1} order".format(i, deg))
                break
            elif i==maxdeg:#if we've reached the max degree tested and still haven't broken the loop, it means none of them fits.
                print("None of the polynomial fit in log-log space is found to extrapolate properly! using the default {0} order fit...".format(deg) )
                p = polyfit(ln(E), ln(eff), deg)
    print("The covariance matrix is", pcov)

    def efficiency_curve(E):
        if type(E) in [uncertainties.core.Variable, uncertainties.core.AffineScalarFunc]:
            lnE = ln(E.n)
        else:
            lnE = ln(E)
        lneff = np.sum( [p[::-1][i]* lnE**i for i in range(len(p))], axis=0) #coefficient_i * x ** i
        
        if cov:
            lnE_powvector = [lnE**i for i in range(len(p))][::-1]
            variance_on_lneff = (lnE_powvector @ pcov @ lnE_powvector) # variance on lneff
            if type(E) in [uncertainties.core.Variable, uncertainties.core.AffineScalarFunc]:
                error_of_lnE = E.s/E.n
                variance_from_E = sum([p[::-1][i]*i*lnE**(i-1) for i in range(1, len(p))])**2 * (error_of_lnE)**2
                variance_on_lneff += variance_from_E
            lneff_variance = exp(lneff)**2 * variance_on_lneff
            return uncertainties.core.Variable( exp(lneff), sqrt(lneff_variance) )
        else:
            return exp(lneff)
    return efficiency_curve

def integrate_count(decay_constant, count_time=COUNT_TIME):
    exponent = -decay_constant*count_time
    if type(exponent) in [uncertainties.core.Variable, uncertainties.core.AffineScalarFunc]:
        exponentiated_term = uncertainties.core.Variable(exp(exponent.n), exp(exponent.n)*exponent.s)
        return 1 - exponentiated_term
    else:
        return 1-exp(exponent)

def add_unc(c):
    return uncertainties.core.Variable(c, sqrt(c))

def compute_weighted_average(quantities):
    for c in quantities:
        assert type(c) in [uncertainties.core.Variable, uncertainties.core.AffineScalarFunc], "Only accept uncertainties.core.Variable/AffineScalarFunc"
    if len(quantities)>0:
        new_variance = 1/ sum([ (1/c.s**2) for c in quantities])
        unnormalized_mean = sum([ (c.n/(c.s**2)) for c in quantities ])
        best_fit_N = uncertainties.core.Variable(unnormalized_mean* new_variance, sqrt(new_variance))
        return best_fit_N
    else:
        return None

def one_mole():
    return 6.02214076E23 # fixed to one mole

def get_min_curve_contr(integrated_apriori):
    max_localization_resistance = max(integrated_apriori)*IRRADIATION_DURATION # unit: cm^4 s^2, i.e. identical to the inverse of square of error on flux.
    # The smaller the value, the stricter it becomes, requiring more parent atoms.
    # Vice versa: the larger the value, the more lenient it becomes, asking for a lower minimum thickness later.
    return max_localization_resistance**(-2) * MIN_CURVE_CONTR_MULTIPLER # minimum contribution to the curvature of chi^2 manifold in phi space. (minimum contribution to ∇χ²)

def serialize(data): #operable on dictionaries.
    if type(data) in [uncertainties.core.Variable, uncertainties.core.AffineScalarFunc]:
        data = str(data.n) + str("+/-")+str(data.s)
    elif type(data)==dict:
        for k,v in data.items():
            data[k] = serialize(v)
    elif type(data)==list: #then it is a class
        data = [serialize(i) for i in data]
    elif isinstance(data, Tabulated1D):
        data = "continuous function (removed)"
    else:
        pass
    return data

def load_R_rr_radiation(working_dir):
    R_file = os.path.join(working_dir, "R.csv")
    rr_file = os.path.join(working_dir, "rr.csv")
    spectra_file = os.path.join(working_dir, "spectra.json")
    
    R = pd.read_csv(R_file, index_col=0)
    rr = pd.read_csv(rr_file, index_col=0)
    with open(spectra_file, 'r') as f:
        spectra_json = json.load(f)
    return R, rr, spectra_json

def R_conversion_main(reaction_and_radiation, apriori_integrated, min_curve_contr, out_dir): # for folding to obtain reaction rates and uncertainties.
    assert os.path.exists(out_dir), "Please create directory {0} to save the output files in first.".format(out_dir)
    R_file, rr_file, spectra_file = os.path.join(out_dir, "R.csv"), os.path.join(out_dir, "rr.csv"), os.path.join(out_dir, "spectra.json")
    spectra_json = {}
    _counted_rr, _R_matrix, r_name_list= [], [], [] # the first tow lists are un-sorted, therefore I prefer to leave them as private variables.

    if SIMULATE_GAMMA_DETECTOR:
        photopeak_efficiency = HPGe_efficiency_curve_generator(out_dir)
    if MAX_OUT_PHOTOPEAK_EFF:
        photopeak_efficiency = lambda x: uncertainties.core.Variable(1, 0)

    for rname, rnr_file in reaction_and_radiation.items():
        if USE_NATURAL_ABUNDANCE:
            try:
                N_target = one_mole() * NATURAL_ABUNDANCE[rname.split('-')[0]]
            except KeyError:
                N_target = 0
                if VERBOSE:
                    print(f"{rname}'s parent is not naturally occurring, and is therefore excluded")
        else:
            N_target = one_mole()
        if N_target !=0: #filtering out the non-naturally occuring elements
            all_peaks = []
            spectra_json[rname] = {} # initialize empty dictionary, since we now know there are non-zero number of this naturally occurring element.
            for dec_files in rnr_file.decay: # for each product (if metastable products form, then it's possible to have two or more products from the same reaction)
                try:# use whichever version that has a more comprehensive (longer) list of gamma radiation
                    len_rad = [len(d.spectra['gamma']['discrete']) for d in dec_files]
                    d = np.argmax(len_rad)
                    dec = dec_files[d]
                except KeyError:
                    dec = dec_files[0]
                # dec contains the decay mode as well, which states the branching ratio and stuff.
                all_peaks_of_this_prod = []
                # for name in ranked_names:
                #     reaction_and_radiation[name].decay[product_num][decay_file_version]
                for rad, specific_spec in dec.spectra.items():
                    if rad in UNWANTED_RADITAION:
                        if VERBOSE:
                            print(f"Warning: unwanted radiation={rad} is found for {rname}. See {spectra_file} for its intensity")
                    if rad=='gamma' or rad=='xray':
                        if specific_spec['continuous_flag']!='discrete':
                            print(f"!! THIS REACTION {rname} HAS A CONTINUOUS ENERGY FLAG FOR ITS {rad}")
                        for peak in specific_spec['discrete']:
                            all_peaks_of_this_prod.append(peak)
                all_peaks.append(all_peaks_of_this_prod)
                spectra_json[rname][dec.nuclide['name']] = dec.spectra.copy()
            #deterministic part:
            # calculate the expected number of nuclides left at the beginning of the measurement
            N_infty = rnr_file.sigma @ apriori_integrated * IRRADIATION_DURATION * N_target *CM2_BARNS_CONVERSION# unit: number of atoms # assuming flash irradiation
            # start to split-up the prediction by products, i.e. if there are multiple products, then we'll have to sum them.
            N_0, decay_correct_factors_with_unc, half_life, decay_constant = [], [], [], []
            for product in rnr_file.decay:
                decay_correct_factors_with_unc.append( total_decay_correct(product, TRANSIT_TIME) ) # correct for flash irradiation assumption, and then for decay during transit.
                decay_constant.append( product[0].decay_constant ) #choose the first version of that product that comes up.
                half_life.append( product[0].half_life )
                N_0.append( N_infty * decay_correct_factors_with_unc[-1].n )# unit: number of atoms
                
            if SIMULATE_GAMMA_DETECTOR: #(Only go into this for loop if we're propagating errors starting from the gamma radiations.')
                #create the following lists once for different product within the same reaction.
                N0_with_unc_of_each_prod = [] #len=len(all_peaks)=j
                for j in range(len(all_peaks)): # for each decay product
                    #create some empty lists, which will have length=len(all_peaks_of_this_prod)=i
                    reconstructed_number_of_decayed_atoms = []
                    for i in range(len(all_peaks[j])): #for each peak
                        peak = all_peaks[j][i]
                        # Sum over the count rate from each peak.
                        eff = geometric_efficiency() * photopeak_efficiency(peak['energy'])
                        #^the error when sampling from different parts of the efficiency curve supposed to be correlated; but I can't be bothered to make it so.
                        intens = peak['intensity']/100

                        initial_count_rate = N_0[j] * intens * eff.n * decay_constant[j]
                        peak['efficiency'] = eff # add in the detection efficiency too!
                        #This works because (peak) is linked to the list (all_peaks)
                        #which is in turn linked to the dictionary (spectra_json) where it is added.
                        
                        # if initial_count_rate.n>=MIN_COUNT_RATE_PER_MOLE_PARENT and #Stop using MIN_COUNT_PER_MOLE_PARENT because it's less useful than MIN_COUNT_PER_MOLE_PARENT
                        if peak['energy'].n==np.clip(peak['energy'].n,*THRESHOLD_ENERGY): # only consider those with well calibrated efficiency
                            counts = N_0[j] * intens.n * eff.n * integrate_count(decay_constant[j].n, COUNT_TIME) #number of gammas counted
                            if counts>MIN_COUNT_PER_MOLE_PARENT:

                                #end of deterministic part; start propagating errors
                                # Now pretend I've finished acquiring these numbers from the HPGe detector, and obtained the associated uncertainties.
                                counts_with_unc = add_unc(counts)
                                reconstructed_number_of_decayed_atoms.append(counts_with_unc / (eff * intens))
                                count_rates_with_unc = counts_with_unc/ integrate_count(decay_constant[j], COUNT_TIME)
                    weighted_average = compute_weighted_average(reconstructed_number_of_decayed_atoms)
                    if weighted_average is not None:
                        N0_with_unc_of_each_prod.append( weighted_average/integrate_count(decay_constant[j], COUNT_TIME) )
                    else:
                        N0_with_unc_of_each_prod.append(None)
                        #maximum length of N0_with_unc_of_each_prod is j=len(all_products)=number of nuclides.
                    
                    # Monkey patch in measurement information dictionary
                    product_j = rnr_file.decay[j][0].nuclide['name']
                    spectra_json[rname][product_j]['measurement'] = {"N_0":N_0[j], # For every mole of parent elements/isotope, you'll have this many left at measurement time
                                            "decay_constant":decay_constant[j], # which will emit radiation at this rate per daughter nuclide.
                                            "half_life": half_life[j],
                                            "decay_correct_factor": decay_correct_factors_with_unc[j], # fraction remaining after transit from beamline/reactor into detector.
                                            "fraction_expected_to_decay": integrate_count(decay_constant[j], COUNT_TIME), # fraction of N_0 expected to decay during the length of the measurement time.
                                            "measurement_time":COUNT_TIME,
                                            "irradiation_time":IRRADIATION_DURATION
                                            }
                    if EVALUATE_THERMAL:
                        spectra_json[rname][product_j]['measurement']['area_pmp'] =  rnr_file.thermal_xs * N_target * CM2_BARNS_CONVERSION
            else: #assume 100% detection efficiency
                error_on_N_0 = sqrt(N_0)*(1/sqrt(COUNT_TIME))
                N0_with_unc_of_each_prod = [uncertainties.core.Variable(N_0[j], error_on_N_0[j]) for j in range(len(N_0))]
            all_products = [productlist[0] for productlist in rnr_file.decay]

            #add the nuclide into the list
            if len(N0_with_unc_of_each_prod)>0 and all([(i is not None) for i in N0_with_unc_of_each_prod]): #only add it if all of its products produces detectible radiation.
                N0_with_unc_of_each_prod = ary(N0_with_unc_of_each_prod)
                N_infty_with_unc = sum(N0_with_unc_of_each_prod/decay_correct_factors_with_unc)/len(N0_with_unc_of_each_prod) # assume identical branching ratios
                # N_infty_with_unc = (N0_with_unc_of_each_prod/decay_correct_factors_with_unc)[0] # assume the majority of it becomes the first product listed
                # N_infty_with_unc = (N0_with_unc_of_each_prod/decay_correct_factors_with_unc)[-1]# assume the majority of it becomes the last product listed
                Rk =  CM2_BARNS_CONVERSION * ary(rnr_file.sigma) * N_target
                min_mole_p = min_curve_contr / Rk.dot(Rk) * N_infty_with_unc.s**2
                _counted_rr.append([N_infty_with_unc.n, N_infty_with_unc.s, min_mole_p])
                _R_matrix.append(Rk)
                if not FOCUS_MODE:
                    print(rname, "->", "|".join([prod.nuclide['name'] for prod in all_products])," is added")
                r_name_list.append(rname)
            elif VERBOSE:
                print(f"Not every product of {rname} has a detectible radiation. Therefore it is excluded.")

            if len(all_products)>1: # take only the first item in each product list as the representative nuclide information dictionary.
                if not FOCUS_MODE:
                    print(rname, "has more than 1 product, namely", [prod.nuclide['name'] for prod in all_products] )
                    print(f"Assuming there is the 100% probability of decaying into all of these products, summing up to {len(all_products)}00% probability of decay...")
                    print("Parent:", rnr_file.parent)
                    print("Daughters are as follows:")
                    for nuc in [prod.nuclide for prod in all_products]:
                        print(nuc)
    R = pd.DataFrame(_R_matrix, index=r_name_list)
    rr = pd.DataFrame(_counted_rr, index=r_name_list, columns=['N_infty per mole parent','uncertainty', 'min mole of parent'])
    # new_rname_order = ary(r_name_list)[np.argsort( _counted_rr)][::-1]
    # R = R.loc[new_rname_order]
    # rr=rr.loc[new_rname_order]

    with open(spectra_file, mode='w', encoding='utf-8') as f:
        spec_copy= copy.deepcopy(spectra_json)
        json.dump(serialize(spec_copy), f)
    print(f"The spectra for each reaction is saved at {spectra_file}")
    
    R.to_csv(R_file, index_label='rname')
    rr.to_csv(rr_file, index_label='rname')
    primer_sentence = "Number of nuclides transmuted after irradiation of "+str(one_mole())+" of that "
    if USE_NATURAL_ABUNDANCE:
        primer_sentence+= "element in its natural composition"
    else:
        primer_sentence+= "isotope"
    print(primer_sentence+"by the un-normalized apriori spectrum for {0}s, along with its measurement uncertainties, is saved in {1}".format(str(IRRADIATION_DURATION), rr_file))
    print(len(rr), "reactions are considered in total.")
    print(R_file, "contains the response matrix with the unit: cm^2")
    print("I.e. The number of transmutation per mole parent of nuclide [k] per cm^-2 of neutron fluence (in the i^th bin) = R[k][i]")

    return R, rr, spectra_json

if __name__=='__main__':
    reaction_and_radiation = read_rnr(sys.argv[-1])
    integrated_apriori = get_integrated_apriori_value_only(sys.argv[-1])
    min_curve_contr = get_min_curve_contr(integrated_apriori) 
    R, rr, spectra_json = R_conversion_main(reaction_and_radiation, integrated_apriori, min_curve_contr, out_dir = sys.argv[-1]) # save only the relevant radiations and rr etc. information out of the reaction_and_radiation dict.

    # Tasks
    '''
    3. Then implement gamma overlap checks, and allow it to be imported into the next file.
    '''
    # Tests:
    '''
    1. The SIMULATE_GAMMA_DETECTOR=True should perform slightly worse (with larger uncertainties, but same means) than SIMULATE_GAMMA_DETECTOR=False
    2. Increasing COUNT_TIME should asymptotically improve the uncertainties to a limit. #DONE
    3. Fast decaying should be affected more by increasing IRRADIATION_DURATION than slow decaying. #DONE
    4. Use the integration function of the Tabulated1D properly. #DONE, see test
    '''