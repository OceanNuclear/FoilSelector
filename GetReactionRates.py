import json
import sys, os, time
from tqdm import tqdm
import openmc
from numpy import array as ary
from numpy import log as ln
from numpy import sqrt
import numpy as np
import uncertainties
from uncertainties.core import Variable
import pandas as pd
from collections import namedtuple, OrderedDict
from misc_library import (  get_apriori,
                            get_fraction,
                            pick_material,
                            unserialize_dict,
                            get_elemental_fractions,
                            save_parameters_as_json,                            
                            extract_elem_from_string,
                            get_natural_average_atomic_mass, 
                            convert_elemental_to_isotopic_fractions,
                            get_average_atomic_mass_from_isotopic_fractions,
                            Bateman_mat_exp_num_decays)

def build_decay_chain(decay_parent, decay_dict, decay_constant_threshold=1E-23):
    """
    Build the entire decay chain for a given starting isotope.

    decay_parent : str
        names of the potentially unstable nuclide
    decay_dict : dictionary
        the entire decay_dict containing all of the that there is.
    """
    if not decay_parent in decay_dict:
        # decay_dict does not contain any decay data record about the specified isotope (decay_parent), meaning it is (possibly) stable.
        return_dict = {'name':decay_parent, 'decay_constant':Variable(0.0,0.0), 'countable_photons':Variable(0.0,0.0), 'modes':{}}
    else: # decay_dict contains the specified decay_parent
        parent = decay_dict[decay_parent]
        return_dict = {'name':decay_parent, 'decay_constant':parent['decay_constant'], 'countable_photons':parent['countable_photons']} # countable_photons per decay of this isotope
        if decay_dict[decay_parent]['decay_constant']<=decay_constant_threshold:
            # if this isotope is rather stable.
            return return_dict
        else:
            return_dict['modes'] = []
            for name, branching_ratio in decay_dict[decay_parent]['branching_ratio'].items():
                if name!=decay_parent:
                    return_dict['modes'].append( {'daughter': build_decay_chain(name, decay_dict), 'branching_ratio': branching_ratio} )
    return return_dict

class IsotopeDecay(namedtuple('IsotopeDecay', ['names', 'branching_ratios', 'decay_constants', 'countable_photons'])):
    """
    Quickly bodged together class that contains 4 attributes:
        names as a list,
        branching_ratios as a list,
        decay_constants as a list,
        countable_photons of the last isotope as a uncertainties.core.AffineFunc object (or a scalar).
    Made so that the __add__ method would behave differently than a normal tuple.
    """
    def __add__(self, subordinate):
        return IsotopeDecay(self.names+subordinate.names, self.branching_ratios+subordinate.branching_ratios, self.decay_constants+subordinate.decay_constants, subordinate.countable_photons)

def linearize_decay_chain(decay_file):
    """
    Return a comprehensive list of path-to-nodes. Each path starts from the origin, and ends at the node of interest.
    Each node in the original graph must be must be recorded as the end node exactly once.
    The initial population is always assumed as 1.
    """
    self_decay = IsotopeDecay([decay_file['name']],
                            [Variable(1.0, 0.0)], # branching ratio dummy value
                            [decay_file['decay_constant']],
                            decay_file['countable_photons'])
    all_chains = [self_decay]
    if 'modes' in decay_file: # expand the decay modes if there are any.
        for mode in decay_file['modes']:
            this_branch = linearize_decay_chain(mode['daughter']) # note that this is a list, so we need to unpack it.
            for subbranch in this_branch:
                subbranch.branching_ratios[0] = mode['branching_ratio']
                all_chains.append(self_decay+subbranch)
    return all_chains # returns a list

def Bateman_num_decays_factorized(branching_ratios, decay_constants, a, b, c, DEBUG=False, decay_constant_threshold=1E-23):
    """
    Calculates the amount of decay radiation measured using the Bateman equation.
    branching_ratio : array
        the list of branching ratios of all its parents, and then itself, in the order that the decay chain was created.
    a : scalar
        End of irradiation period.
        Irrdiation power = 1/a, so that total amount of irradiation = 1.
    b : scalar
        Start of gamma measurement time.
    c : scalar
        End of gamma measurement time.
    The initial population is always assumed as 1.
    Reduce decay_constant_threshold when calculating on very short timescales.
    """
    # assert len(branching_ratios)==len(decay_constants), "Both are lists of parameters describing the entire pathway of the decay chain up to that isotope."
    if any([i<=decay_constant_threshold for i in decay_constants]):
        if DEBUG: print(f"{decay_constants=} \ntoo small, escaped.")
        return Variable(0.0, 0.0) # catch the cases where there are zeros in the decay_rates
            # in practice this should only happen for chains with stable parents, i.e.
            # this if-condition would only be used if the decay chain is of length==1.    
    premultiplying_factor = np.product(decay_constants[:])*np.product(branching_ratios[1:])/a
    inverted_matrix = np.diff(np.meshgrid(decay_constants, decay_constants)[::-1], axis=0)[0]
    # inverted_matrix += np.diag(decay_constants)**2
    inverted_matrix += np.diag(np.ones(len(decay_constants)))
    final_matrix = 1/inverted_matrix
    multiplying_factors = np.product(final_matrix, axis=-1)
    try:
        vector_uncollapsed = ary([1/ary(decay_constants)**2
                        *uncertainties.unumpy.expm1(ary(decay_constants)*a)
                        *uncertainties.unumpy.expm1(ary(decay_constants)*(c-b))
                        *uncertainties.unumpy.exp(-ary(decay_constants)*c) ], dtype=object)
    except OverflowError:
        # remove the element that causes such an error from the chain, and recompute
        decay_constants_copy = decay_constants.copy()
        for ind, l in list(enumerate(decay_constants))[::-1]:
            try:
                ary([1/ary(l)**2
                        *uncertainties.unumpy.expm1(ary(l)*a)
                        *uncertainties.unumpy.expm1(ary(l)*(c-b))
                        *uncertainties.unumpy.exp(-ary(l)*c) ])
            except OverflowError: # catch all cases and pop them out of the list
                decay_constants_copy.pop(ind)

        return Bateman_num_decays_factorized(branching_ratios, decay_constants_copy, a, b, c, DEBUG, decay_constant_threshold)
    vector = np.sum(vector_uncollapsed, axis=0)
    if DEBUG:
        print(#inverted_matrix, '\n', final_matrix, '\n',
        'multiplying_factors=\n', multiplying_factors,
        #'\n--------\nvector=\n', vector,
        )
        try:
            print("Convolution term = \n", uncertainties.unumpy.expm1(ary(decay_constants)*a)/ary(decay_constants))
            print("measurement (integration) term\n", uncertainties.unumpy.expm1(ary(decay_constants)*(c-b))/ary(decay_constants))
            print("end of measurement term\n", uncertainties.unumpy.exp(-ary(decay_constants)*c))
        except:
            print("Overflow error")
    return premultiplying_factor * (multiplying_factors @ vector)

def check_differences(decay_constants):
    difference_matrix = np.diff(np.meshgrid(decay_constants, decay_constants)[::-1], axis=0)[0]
    return abs(difference_matrix)[np.triu_indices(len(decay_constants), 1)]

def tprint(msg):
    time_mark = time.time() - prog_start_time
    print("\n[t=+{:2.2f}s]".format(time_mark)+msg)

# PARAMETERS
IRRADIATION_DURATION = 3600 # spread the irradiation power over the entire course of this length
TRANSIT_DURATION = 5*60
MEASUREMENT_DURATION = 3600
FOIL_AREA = 10 # cm^2
ENRICH_TO_100_PERCENT = False # allowing enrichment means 100% of that element being made of the specified isotope only

BARN = 1E-24
MM_CM = 0.1

PHYSICAL_PROP_FILE = ".physical_parameters/elemental_frac_isotopic_frac_physical_property.csv"
CONDENSED_DECAY_INFO_FILE = 'decay_counts.json'

if __name__=='__main__':
    prog_start_time = time.time()
    assert os.path.exists(PHYSICAL_PROP_FILE), "Expected physical property file at ./{}".format(os.path.relpath(PHYSICAL_PROP_FILE))
    apriori_flux, apriori_fluence = get_apriori(sys.argv[-1], IRRADIATION_DURATION)

    tprint(f"Reading ./{os.path.relpath(PHYSICAL_PROP_FILE)} to be interpreted as the physical parameters.")
    physical_prop = pd.read_csv(PHYSICAL_PROP_FILE, index_col=[0])

    with open(os.path.join(sys.argv[-1], CONDENSED_DECAY_INFO_FILE), 'r') as f:
        decay_dict = json.load(f)
        decay_dict = unserialize_dict(decay_dict)
    sigma_df = pd.read_csv(os.path.join(sys.argv[-1], 'response.csv'), index_col=[0])

    detected_counts_per_primary_product, total_counts_per_primary_product = {}, {}
    tprint("Calculating the expected number of photopeak counts for each type of product created:")
    product_set = set([parent_product_mt.split("-")[1] for parent_product_mt in sigma_df.index])
    for product in tqdm(product_set):
        detected_counts_per_primary_product[product] = [{
                    'pathway': '-'.join(subchain.names),
                            # (# of photons detected per nuclide n decayed) = (# of photons detected per decay of nuclide n) * lambda_n * \int(population)dT
                    # 'counts':Bateman_num_decays_factorized(subchain.branching_ratios, subchain.decay_constants,
                    'counts':np.product(subchain.branching_ratios[1:])*
                            Bateman_mat_exp_num_decays(subchain.decay_constants, 
                            IRRADIATION_DURATION,
                            IRRADIATION_DURATION+TRANSIT_DURATION,
                            IRRADIATION_DURATION+TRANSIT_DURATION+MEASUREMENT_DURATION
                            )*subchain.countable_photons,
                    } for subchain in linearize_decay_chain(build_decay_chain(product, decay_dict))]
        total_counts_per_primary_product[product] = sum([path["counts"] for path in detected_counts_per_primary_product[product]])
    # get the production rate for each reaction
    population = pd.DataFrame({'production of primary product per parent atom':(sigma_df.values*BARN) @ apriori_fluence}, index=sigma_df.index)
    # add the total counts of gamma photons detectable per primary product column
    tprint("Matching the reactions to their decay product count...")
    population['total gamma counts per primary product'] = [total_counts_per_primary_product[parent_product_mt.split("-")[1]] for parent_product_mt in sigma_df.index]
    # add the final counts accumulated per parent atom column
    population['final counts accumulated per parent atom'] = population['total gamma counts per primary product'] * population['production of primary product per parent atom']
    # sort by activity and remove all nans
    tprint("Re-ordering the dataframe by the final counts accumulated per parent atom and removing the entries with zero counts...")
    population.sort_values('final counts accumulated per parent atom', inplace=True, ascending=False)
    population = population[population['total gamma counts per primary product']>0.0] # keeping only those with positive counts.
    # select the default materials and get its relevant parameters
    default_material, partial_number_density = [], []

    tprint("Selecting the default material to be used:")
    for parent_product_mt in tqdm(population.index):
        parent = parent_product_mt.split('-')[0]
        if parent[len(extract_elem_from_string(parent)):]=='0': # take care of the species which are a MIXED natural composition of materials, e.g. Gd0
            parent = parent[:-1]
        if ENRICH_TO_100_PERCENT: # allowing enrichment means 100% of that element being made of the specified isotope only
            parent = extract_elem_from_string(parent)
        if parent not in physical_prop.columns:
            # if there isn't a parent material
            default_material.append('Missing (N/A)')
            partial_number_density.append(0.0)
            continue
        material_info = pick_material(parent, physical_prop)
        default_material.append(material_info.name+" ("+material_info['Formula']+")")
        partial_number_density.append(material_info['Number density-cm-3'] * material_info[parent])

    population["default material"] = default_material
    population["partial number density (cm^-3)"] = partial_number_density
    population["gamma counts per volume of foil (cm^-3)"] = population["final counts accumulated per parent atom"] * population["partial number density (cm^-3)"]
    population["gamma counts per unit thickness of foil (mm^-1)"] = population["gamma counts per volume of foil (cm^-3)"] * FOIL_AREA * MM_CM# assuming the area = Foil Area
    tprint("Re-ordering the dataframe according to the counts per volume...")
    population.sort_values("gamma counts per volume of foil (cm^-3)", inplace=True, ascending=False) # sort again, this time according to the required volume
    tprint("Removing all reactions whose products is the same as the product (i.e. elastic scattering):")
    population = population[ary([parent_product_mt.split("-")[0] != parent_product_mt.split("-")[1] for parent_product_mt in population.index])]

    tprint("Saving as 'counts.csv'...")
    population.to_csv(os.path.join(sys.argv[-1], 'counts.csv'), index_label='rname')

    # save parameters at the end.
    save_parameters_as_json(sys.argv[-1], dict(
        IRRADIATION_DURATION=IRRADIATION_DURATION,
        TRANSIT_DURATION=TRANSIT_DURATION,
        MEASUREMENT_DURATION=MEASUREMENT_DURATION,
        FOIL_AREA=FOIL_AREA,
        ENRICH_TO_100_PERCENT=ENRICH_TO_100_PERCENT,
        PHYSICAL_PROP_FILE = PHYSICAL_PROP_FILE,
        CONDENSED_DECAY_INFO_FILE = CONDENSED_DECAY_INFO_FILE,
        )
    )
    tprint("Run complete. See results in 'counts.csv'.")