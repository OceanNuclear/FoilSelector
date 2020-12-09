import openmc
from numpy import array as ary
from numpy import log as ln
from numpy import sqrt
import numpy as np
if (DEV_MODE:=False):
    from matplotlib import pyplot as plt
import uncertainties
from uncertainties.core import Variable
import json
import sys, os
import pandas as pd
from collections import namedtuple, OrderedDict
from misc_library import haskey
import math

def kahan_sum(a, axis=0):
    """
    Carefully add together sum to avoid floating point precision problem.
    Retrieved (and then modified) from
    https://github.com/numpy/numpy/issues/8786
    """
    s = np.zeros(a.shape[:axis] + a.shape[axis+1:])
    c = np.zeros(s.shape)
    for i in range(a.shape[axis]):
        # http://stackoverflow.com/42817610/353337
        y = a[(slice(None),) * axis + (i,)] - c
        t = s + y
        c = (t - s) - y
    return t

def incremental_sum(unsorted_list):
    """
    Calculate the sum
    unsorted_list : list
    """
    order = np.argsort([abs(i) for i in unsorted_list])
    return _sum(*ary(unsorted_list)[order])

def _sum(*list_of_scalars):
    if len(list_of_scalars)>1:
        return _sum(list_of_scalars[0]+list_of_scalars[1], *list_of_scalars[2:])
    elif len(list_of_scalars)==1:
        return list_of_scalars[0]
    else:
        return 0.0

def _product(*list_of_scalars):
    if len(list_of_scalars)>1:
        return _product(list_of_scalars[0]*list_of_scalars[1], *list_of_scalars[2:])
    elif len(list_of_scalars)==1:
        return list_of_scalars[0]
    else:
        return 1.0

def unserialize_dict(mixed_object):
    """
    Turn the string representation of the uncertainties back into uncertainties.core.Variable 's.
    """
    if isinstance(mixed_object, dict):
        for key, val in mixed_object.items():
            mixed_object[key] = unserialize_dict(val)
    elif isinstance(mixed_object, list):
        for ind, item in enumerate(mixed_object):
            mixed_object[ind] = unserialize_dict(item)
    elif isinstance(mixed_object, str):
        if '+/-' in mixed_object: # is an uncertainties.core.Variable object
            mixed_object.split
            if ')' in mixed_object:
                multiplier = float('1'+mixed_object.split(')')[1])
                mixed_object_stripped = mixed_object.split(')')[0].strip('(')
            else:
                multiplier = 1.0
                mixed_object_stripped = mixed_object
            mixed_object = Variable(*[float(i)*multiplier for i in mixed_object_stripped.split('+/-')])
        else:
            pass # just a normal string
    else: # unknown type
        pass
    return mixed_object

def build_decay_chain(decay_parent, decay_dict, decay_constant_threshold=1E-23):
    """
    Build the entire decay chain for a given starting isotope.

    decay_parent : str
        names of the potentially unstable nuclide
    decay_dict : dictionary
        the entire decay_dict containing all of the that there is.
    """
    if not haskey(decay_dict, decay_parent):
        return_dict = {'name':decay_parent, 'decay_constant':Variable(0.0,0.0), 'countable_photons':Variable(0.0,0.0)}
        return return_dict
    elif decay_dict[decay_parent]['decay_constant']<=decay_constant_threshold:
        parent = decay_dict[decay_parent]
        return_dict = {'name':decay_parent, 'decay_constant':parent['decay_constant'], 'countable_photons':parent['countable_photons']}
        return return_dict
    else:
        parent = decay_dict[decay_parent]
        return_dict = {'name':decay_parent, 'decay_constant':parent['decay_constant'], 'countable_photons':parent['countable_photons']}
        return_dict['modes'] = [{'daughter':build_decay_chain(mode['daughter'], decay_dict), 'branching_ratio':mode['branching_ratio']} for mode in parent['modes'] if mode['daughter']!=decay_parent] # prevent infinite recursion
        return return_dict

class IsotopeDecay(namedtuple('IsotopeDecay', ['names', 'branching_ratios', 'decay_constants', 'countable_photons'])):
    """
    Quickly bodged together class that contains 3 attributes:
        names as a list,
        branching_ratios as a list,
        decay_constants as a list.
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
    if haskey(decay_file, 'modes'):
        for mode in decay_file['modes']:
            this_branch = linearize_decay_chain(mode['daughter']) # note that this is a list, so we need to unpack it.
            for subbranch in this_branch:
                subbranch.branching_ratios[0] = mode['branching_ratio']
                all_chains.append(self_decay+subbranch)
    return all_chains # returns a list

def Bateman_equation_generator(branching_ratios, decay_constants, decay_constant_threshold=1E-23):
    """
    Calculates the radioisotope population at the end of the chain specified.
    branching_ratio : array
        the list of branching ratios of all its parents, and then itself, in the order that the decay chain was created.
    a : scalar
        End of irradiation period.
        Irradiation schedule = from 0 to a, 
        with rrdiation power = 1/a, so that total amount of irradiation = 1.0.
    """
    # assert len(branching_ratios)==len(decay_constants), "Both are lists of parameters describing the entire pathway of the decay cahin up to that isotope."
    if any([i<=decay_constant_threshold for i in decay_constants]):
        return lambda x: x-x # catch the cases where there are zeros in the decay_rates
    premultiplying_factor = _product(*decay_constants[:-1])*_product(*branching_ratios[1:])
    inverted_matrix = np.diff(np.meshgrid(decay_constants, decay_constants)[::-1], axis=0)
    inverted_matrix += np.diag(np.ones(len(decay_constants)))
    final_matrix = 1/inverted_matrix
    multiplying_factors = np.product(final_matrix, axis=-1)
    def calculate_population(t):
        vector = (uncertainties.unumpy.exp(-ary(decay_constants)*max(0,t))
                # -uncertainties.unumpy.exp(-ary(decay_constants)*(c-a))
                # -uncertainties.unumpy.exp(-ary(decay_constants)*b)
                # +uncertainties.unumpy.exp(-ary(decay_constants)*(b-a))
                )
        return premultiplying_factor * (multiplying_factors @ vector)
    return calculate_population

def Bateman_convolved_generator(branching_ratios, decay_constants, a, decay_constant_threshold=1E-23):
    if any([i<=decay_constant_threshold for i in decay_constants]):
        return lambda x: x-x # catch the cases where there are zeros in the decay rates.
    premultiplying_factor = _product(*decay_constants[:-1])*_product(*branching_ratios[1:])/a
    inverted_matrix = np.diff(np.meshgrid(decay_constants, decay_constants)[::-1], axis=0)
    inverted_matrix += np.diag(decay_constants)
    final_matrix = 1/inverted_matrix
    multiplying_factors = np.product(final_matrix, axis=-1)
    def calculate_convoled_population(t):
        """
        Calculates the population at any given time t when a non-flash irradiation schedule is used,
        generated using irradiation duration a={} seconds
        """.format(a)
        vector_uncollapsed = ary([+uncertainties.unumpy.exp(-ary([l*np.clip(t-a,0, None) for l in decay_constants])), -uncertainties.unumpy.exp(-ary([l*np.clip(t,0, None) for l in decay_constants]))], dtype=object)
        vector = kahan_sum(vector_uncollapsed, axis=0)
        return premultiplying_factor*(multiplying_factors@vector)
    return calculate_convoled_population

def Bateman_integrated_population(branching_ratios, decay_constants, a, b, c, DEBUG=False, decay_constant_threshold=1E-23):
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
    """
    # assert len(branching_ratios)==len(decay_constants), "Both are lists of parameters describing the entire pathway of the decay chain up to that isotope."
    if any([i<=decay_constant_threshold for i in decay_constants]):
        if DEBUG: print(f"{decay_constants=} \ntoo small, escaped.")
        return Variable(0.0, 0.0) # catch the cases where there are zeros in the decay_rates
            # in practice this should only happen for chains with stable parents, i.e.
            # this if-condition would only be used if the decay chain is of length==1.    
    premultiplying_factor = _product(*decay_constants[:-1])*_product(*branching_ratios[1:])/a
    inverted_matrix = np.diff(np.meshgrid(decay_constants, decay_constants)[::-1], axis=0)[0]
    inverted_matrix += np.diag(decay_constants)**2
    final_matrix = 1/inverted_matrix
    multiplying_factors = np.product(final_matrix, axis=-1)
    vector_uncollapsed = ary([  +uncertainties.unumpy.exp(-ary(decay_constants)*c),
                    -uncertainties.unumpy.exp(-ary(decay_constants)*(c-a)),
                    -uncertainties.unumpy.exp(-ary(decay_constants)*b),
                    +uncertainties.unumpy.exp(-ary(decay_constants)*(b-a),)], dtype=object)
    vector = kahan_sum(vector_uncollapsed, axis=0)
    if DEBUG:
        print(#inverted_matrix, '\n', final_matrix, '\n',
        'multiplying_factors=\n', multiplying_factors,
        #'\n--------\nvector=\n', vector,
        )
        print("Vector left = \n", uncertainties.unumpy.exp(-ary(decay_constants)*(b-a)) - uncertainties.unumpy.exp(-ary(decay_constants)*b))
        print("Vector right= \n", uncertainties.unumpy.exp(-ary(decay_constants)*(c-a)) - uncertainties.unumpy.exp(-ary(decay_constants)*c))
    return premultiplying_factor * (multiplying_factors @ vector)

# PARAMETERS
IRRADIATION_DURATION = 3600
TRANSIT_DURATION = 5*60
MEASUREMENT_DURATION = 3600

if __name__=='__main__':
    assert os.path.exists(os.path.join(sys.argv[-1], 'integrated_apriori.csv')), "Output directory must already have integrated_apriori.csv for calculating the radionuclide populations."
    apriori = pd.read_csv(os.path.join(sys.argv[-1], 'integrated_apriori.csv'))['value'].values # integrated_apriori.csv is a csv with header = value and number of rows = len(gs); no index.
    # print("Reading integrated_apriori.csv as the flux, i.e. total number of neutrons/cm^2/eV/s, averaged over the IRRADIATION_DURATION = {}".format(IRRADIATION_DURATION))

    with open(os.path.join(sys.argv[-1], 'decay_records.json'), 'r') as f:
        decay_dict = json.load(f)
        decay_dict = unserialize_dict(decay_dict)
    sigma_df = pd.read_csv(os.path.join(sys.argv[-1], 'response.csv'), index_col=[0])

    detected_counts_per_parent_material_nuclide = {}
    for parent_product_mt in sigma_df.index:
        product = parent_product_mt.split('-')[1]
        detected_counts_per_parent_material_nuclide[parent_product_mt] = [{
                    'pathway': '-'.join(chain.names),
                    'counts':Bateman_integrated_population(chain.branching_ratios, chain.decay_constants,
                            IRRADIATION_DURATION,
                            IRRADIATION_DURATION+TRANSIT_DURATION,
                            IRRADIATION_DURATION+TRANSIT_DURATION+MEASUREMENT_DURATION,
                            # DEBUG=True,
                            )*chain.countable_photons*chain.decay_constants[-1], # question: do I have to multiply by that at all?
                    } for chain in linearize_decay_chain(build_decay_chain(product, decay_dict))]
    """
    # print all negative cases
    debug_dict, not_debug_dict = {}, {}
    for reaction in detected_counts_per_parent_material_nuclide.values():
        for path in reaction:
            # if len(path['pathway'].split('-'))==2:
            if path['counts']<=0:
                # print(path['counts'], '\t\t', path['pathway'])
                debug_dict[path['pathway']] = path['counts']
            else:
                not_debug_dict[path['pathway']] = path['counts']
    sort_order = np.argsort(list(debug_dict.values()))
    debug_dict = OrderedDict([(key, val) for (key, val) in ary(list(debug_dict.items()))[sort_order]])
    """
    population = pd.DataFrame({'production rate':sigma_df.values @ apriori}, index=sigma_df.index)
    population['total counts per atom per second irradiation'] = [sum([path['counts'] for path in detected_counts_per_parent_material_nuclide[i]]) for i in sigma_df.index]
    population['final counts accumulated'] = population['total counts per atom per second irradiation'] * population['production rate']
    population.sort_values('final counts accumulated', inplace=True, ascending=False)
    population.to_csv(os.path.join(sys.argv[-1], 'raw_counts_per_atom.csv'), index_label='rname')