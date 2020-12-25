import numpy as np
import uncertainties
from uncertainties.core import Variable
import pandas as pd
import os, sys, json
import openmc
from numpy import sqrt
from numpy import log as ln
from tqdm import tqdm
MeV = 1E6
# debugging tools
sdir = lambda x: [i for i in dir(x) if '__' not in i]
import matplotlib.pyplot as plt
plot_tab = lambda tab: plt.plot(*[getattr(tab, ax) for ax in 'xy'])
haskey = lambda dict_instance, key: key in dict_instance.keys()

######################### frequently referenced table for interpreting #################################
from openmc.data.reaction import REACTION_NAME
from openmc.data.endf import SUM_RULES
from openmc.data import NATURAL_ABUNDANCE, atomic_mass, ATOMIC_NUMBER

MT_to_nuc_num = {
    2:(0, 0),
    4:(0, 0),
    11:(-1, -3),
    16:(0, -1),
    17:(0, -2),
    22:(-2, -4),
    23:(-6, -12),
    24:(-2, -5),
    25:(-2, -6),
    28:(-1, -1),
    29:(-4, -8),
    30:(-4, -9),
    32:(-1, -2),
    33:(-1, -3),
    34:(-2, -3),
    35:(-5, -10),
    36:(-5, -11),
    37:(0, -3),
    41:(-1, -2),
    42:(-1, -3),
    44:(-2, -2),
    45:(-3, -5),
    102:(0, 1),
    103:(-1, 0),
    104:(-1, -1),
    105:(-1, -2),
    106:(-2, -2),
    107:(-2, -3),
    108:(-4, -7),
    109:(-6, -11),
    111:(-2, -1),
    112:(-3, -4),
    113:(-5, -10),
    114:(-5, -9),
    115:(-2, -2),
    116:(-2, -3),
    117:(-3, -5),
    152:( 0, -4),
    153:( 0, -5),
    154:(-1, -4),
    155:(-3, -6),
    156:(-1, -4),
    157:(-1, -4),
    158:(-3, -6),
    159:(-3, -6),
    160:( 0, -6),
    161:( 0, -7),
    162:(-1, -5),
    163:(-1, -6),
    164:(-1, -7),
    165:(-2, -7),
    166:(-2, -8),
    167:(-2, -9),
    168:( -2, -10),
    169:(-1, -5),
    170:(-1, -6),
    171:(-1, -7),
    172:(-1, -5),
    173:(-1, -6),
    174:(-1, -7),
    175:(-1, -8),
    176:(-2, -4),
    177:(-2, -5),
    178:(-2, -6),
    179:(-2, -4),
    180:(-4, -10),
    181:(-3, -7),
    182:(-2, -4),
    183:(-2, -3),
    184:(-2, -4),
    185:(-2, -5),
    186:(-3, -4),
    187:(-3, -5),
    188:(-3, -6),
    189:(-3, -7),
    190:(-2, -3),
    191:(-3, -3),
    192:(-3, -4),
    193:(-4, -6),
    194:(-2, -5),
    195:( -4, -11),
    196:(-3, -8),
    197:(-3, -2),
    198:(-3, -3),
    199:(-4, -8),
    200:(-2, -6),
    800:(-2, -3),
    801:(-2, -3)
}
for i in range(51, 92): # (z,n?)
    MT_to_nuc_num[i] = tuple([*MT_to_nuc_num[2], i-50])
for i in range(600, 649): # (z, p?)
    MT_to_nuc_num[i] = tuple([*MT_to_nuc_num[103], i-600])
MT_to_nuc_num[649] = MT_to_nuc_num[103]
for i in range(650, 699): # (z, d?)
    MT_to_nuc_num[i] = tuple([*MT_to_nuc_num[104], i-650])
MT_to_nuc_num[699] = MT_to_nuc_num[104]
for i in range(700, 749): # (z, t?)
    MT_to_nuc_num[i] = tuple([*MT_to_nuc_num[105], i-700])
MT_to_nuc_num[749] = MT_to_nuc_num[105]
for i in range(750, 799): # (z, He3?)
    MT_to_nuc_num[i] = tuple([*MT_to_nuc_num[106], i-750])
MT_to_nuc_num[799] = MT_to_nuc_num[106]
for i in range(800, 849):
    MT_to_nuc_num[i] = tuple([*MT_to_nuc_num[107], i-800])
MT_to_nuc_num[849] = MT_to_nuc_num[107]
for i in range(875, 891):
    MT_to_nuc_num[i] = tuple([*MT_to_nuc_num[16], i-875])
MT_to_nuc_num[891] = MT_to_nuc_num[16]

####################### functions used for reading and saving data######################################
def load_endf_directories(folder_list):
    try:
        assert len(folder_list)>0
        #Please update your python to 3.6 or later.
        print(f"Reading from {len(folder_list)} folders,")
        endf_file_list = []
        for folder in folder_list:
            endf_file_list += [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.endf') or file.endswith('.asc')]
    except (IndexError, AssertionError):
        print("usage:")
        print("'python "+sys.argv[0]+" folders/ containing/ endf/ files/ in/ descending/ order/ of/ priority/ [output/]'")
        print("where the outputs-saving directory 'output/' is only requried when read_apriori_and_gs_df is used.")
        print("Use wildcard (*) to replace directory as .../*/ if necessary")
        print("The endf files in question can be downloaded from online sources")
        # by running the make file.")
        # from https://www.oecd-nea.org/dbforms/data/eva/evatapes/eaf_2010/ and https://www-nds.iaea.org/IRDFF/
        # Currently does not support reading h5 files yet, because openmc does not support reading ace/converting endf into hdf5/reading endf using NJOY yet
        sys.exit()

    print(f"Found {len(endf_file_list)} regular files ending in '.endf' or '.asc'. Reading them ...")
    #read in each file:
    endf_data = []
    for path in tqdm(endf_file_list):
        try:
            endf_data += openmc.data.get_evaluations(path)#works with IRDFF/IRDFFII.endf and EAF/*.endf
        except:
            endf_data += [openmc.data.Evaluation(path),] # works with decay/decay_2012/*.endf
    return endf_data

def detabulate(tabulated_object):
    """
    Convert a openmc.data.
    """
    x = tabulated_object.x.tolist()
    y = tabulated_object.y.tolist()
    interpolation = []
    for ind in range(1,len(x)):
        mask = ind<tabulated_object.breakpoints # may need a little change because interpolation is supposed to have length = len(x)-1
        interpolation.append( tabulated_object.interpolation[mask][0] ) # choose the leftmost section that has energy < the next breakpoint, and add that section's corresponding interp scheme.
    return dict(x=x, y=y, interpolation=interpolation)

class EncoderOpenMC(json.JSONEncoder):
    def default(self, o):
        """
        The object will be sent to the .default() method if it can't be handled
        by the native ,encode() method.
        The original JSONEncoder.default method only raises a TypeError;
        so in this class we'll make sure it handles these specific cases (numpy, openmc and uncertainties)
        before defaulting to the JSONEncoder.default's error raising.
        """
        # numpy types
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.float):
            return float(o)
        # uncertainties types
        elif isinstance(o, uncertainties.core.AffineScalarFunc):
            return str(o)
        # openmc
        elif isinstance(o, openmc.data.Tabulated1D):
            return detabulate(o)
        # return to default error
        else:
            return super().default(o)

class DecoderOpenMC(json.JSONDecoder):
    def decode(self, o):
        """
        Catch the uncertainties
        Edit: This class doesn't work because the cpython/decoder.py is not written in a open-for-expansion principle.
        I suspect this is because turning variables (which aren't strings) into strings is an
            imporper way to use jsons; but I don't have any other solutions for EncoderOpenMC.

        In any case, DecoderOpenMC will be replaced by unserialize_dict below.
        """
        if '+/-' in o:
            if ')' in o:
                multiplier = float('1'+o.split(')')[1])
                o = o.split(')')[0].strip('(')
            else:
                multiplier = 1.0
            return Variable(*[float(i)*multiplier for i in o.split('+/-')])
        else:
            return super().decode(o)

def serialize_dict(mixed_object):
    """
    Deprecated as its functionality is entirely covered by EncoderOpenMC.
    """
    if isinstance(mixed_object, dict):
        for key, val in mixed_object.items():
            mixed_object[key] = serialize_dict(val)
    elif isinstance(mixed_object, list):
        for ind, item in enumerate(mixed_object):
            mixed_object[ind] = serialize_dict(item)
    elif isinstance(mixed_object, uncertainties.core.AffineScalarFunc):
        mixed_object = str(mixed_object) # rewrite into str format
    elif isinstance(mixed_object, openmc.data.Tabulated1D):
        mixed_object = detabulate(mixed_object)
    else: # can't break it down, and probably is a scalar.
        pass
    return mixed_object

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

############################### HPGe Efficiency reader #################################################
def exp(numbers):
    if isinstance(numbers, uncertainties.core.AffineScalarFunc):
        return uncertainties.unumpy.exp(numbers)
    else:
        return np.exp(numbers)

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
    import glob
    file_location = glob.glob(os.path.join(working_dir, '*photopeak_efficiency*.csv'))[0]
    datapoints = pd.read_csv(file_location)
    assert "energ" in datapoints.columns[0].lower(), "The file must contain a header. Energy (eV/MeV) has to be placed in the first column"
    E, eff = datapoints.values.T[:2]
    if 'MeV' in datapoints.columns[0] or 'MeV' in file_location:
        E = MeV*E
    print("Using the gamma detection efficiency curve read from {}".format(file_location))
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
    if cov: 
        print("The covariance matrix is\n", pcov)

    def efficiency_curve(E):
        if isinstance(E, uncertainties.core.AffineScalarFunc):
            lnE = ln(E.n)
        else:
            lnE = ln(E)
        lneff = np.sum( [p[::-1][i]* lnE**i for i in range(len(p))], axis=0) #coefficient_i * x ** i
        
        if cov:
            lnE_powvector = [lnE**i for i in range(len(p))][::-1]
            variance_on_lneff = (lnE_powvector @ pcov @ lnE_powvector) # variance on lneff
            if isinstance(E, uncertainties.core.AffineScalarFunc):
                error_of_lnE = E.s/E.n
                variance_from_E = sum([p[::-1][i]*i*lnE**(i-1) for i in range(1, len(p))])**2 * (error_of_lnE)**2
                variance_on_lneff += variance_from_E
            lneff_variance = exp(lneff)**2 * variance_on_lneff
            return uncertainties.core.Variable( exp(lneff), sqrt(lneff_variance) )
        else:
            return exp(lneff)
    return efficiency_curve

############################## Bateman equation used only for error checking ###########################
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

def Bateman_equation_generator(branching_ratios, decay_constants, decay_constant_threshold=1E-23):
    """
    Calculates the radioisotope population at the end of the chain specified.
    branching_ratio : array
        the list of branching ratios of all its parents, and then itself, in the order that the decay chain was created.
    a : scalar
        End of irradiation period.
        Irradiation schedule = from 0 to a, 
        with rrdiation power = 1/a, so that total amount of irradiation = 1.0.
    Reduce decay_constant_threshold when calculating on very short timescales.
    """
    # assert len(branching_ratios)==len(decay_constants), "Both are lists of parameters describing the entire pathway of the decay cahin up to that isotope."
    if any([i<=decay_constant_threshold for i in decay_constants]):
        return lambda x: x-x # catch the cases where there are zeros in the decay_rates
    premultiplying_factor = np.product(decay_constants[:-1])*np.product(branching_ratios[1:])
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
    premultiplying_factor = np.product(decay_constants[:-1])*np.product(branching_ratios[1:])/a
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
        vector = np.sum(vector_uncollapsed, axis=0)
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
    Reduce decay_constant_threshold when calculating on very short timescales.
    """
    # assert len(branching_ratios)==len(decay_constants), "Both are lists of parameters describing the entire pathway of the decay chain up to that isotope."
    if any([i<=decay_constant_threshold for i in decay_constants]):
        if DEBUG: print(f"{decay_constants=} \ntoo small, escaped.")
        return Variable(0.0, 0.0) # catch the cases where there are zeros in the decay_rates
            # in practice this should only happen for chains with stable parents, i.e.
            # this if-condition would only be used if the decay chain is of length==1.    
    premultiplying_factor = np.product(decay_constants[:-1])*np.product(branching_ratios[1:])/a
    inverted_matrix = np.diff(np.meshgrid(decay_constants, decay_constants)[::-1], axis=0)[0]
    inverted_matrix += np.diag(decay_constants)**2
    final_matrix = 1/inverted_matrix
    multiplying_factors = np.product(final_matrix, axis=-1)
    vector_uncollapsed = ary([  +uncertainties.unumpy.exp(-ary(decay_constants)*c),
                    -uncertainties.unumpy.exp(-ary(decay_constants)*(c-a)),
                    -uncertainties.unumpy.exp(-ary(decay_constants)*b),
                    +uncertainties.unumpy.exp(-ary(decay_constants)*(b-a),)], dtype=object)
    vector = np.sum(vector_uncollapsed, axis=0)
    if DEBUG:
        print(#inverted_matrix, '\n', final_matrix, '\n',
        'multiplying_factors=\n', multiplying_factors,
        #'\n--------\nvector=\n', vector,
        )
        print("Vector left = \n", uncertainties.unumpy.exp(-ary(decay_constants)*(b-a)) - uncertainties.unumpy.exp(-ary(decay_constants)*b))
        print("Vector right= \n", uncertainties.unumpy.exp(-ary(decay_constants)*(c-a)) - uncertainties.unumpy.exp(-ary(decay_constants)*c))
    return premultiplying_factor * (multiplying_factors @ vector)

def expand_bracket(solid):
    """
    expand the compositions of an sub-alloy inside the bracket
    """
    if len(left_bracket_split:=solid.split("("))>1:
        in_brackets, behind_brackets = left_bracket_split[1].split(")")
        # find the text inside the bracket and the NUMBER that immediately follow the bracket.
        fraction_of_bracketed = ""

        for char in behind_brackets: # we will be looping through a COPY of the behind_brackets, so we won't have to worry about mutating the string while looping.
            if char.isnumeric() or char==".":
                fraction_of_bracketed += char
            if char == "/":
                break # stop when it hits a '/' just in case there are more components behind it.
            behind_brackets = behind_brackets[1:] # that character is considered "Used", and is removed from the behind_brackets variable.
        fraction_of_bracketed = float(fraction_of_bracketed)

        modified_in_bracket = ""
        num_compositions = len(in_brackets.split("+"))
        for composition in in_brackets.split("+"):
            modified_in_bracket += composition
            modified_in_bracket += str(fraction_of_bracketed/num_compositions)
            modified_in_bracket += "/"
        modified_in_bracket = modified_in_bracket.rstrip('/') #remove the extranous "/"
        
        return "".join([left_bracket_split[0], modified_in_bracket, behind_brackets])
    else:
        return solid

### Text processing functions solely for supporting the physical properties inquiring functions below ##
def split_at_cap(string):
    """Cut a string up into chunks;
    each chunk of the string contains a single element and its fraction,
    which is expected to start with a capital letter.
    """
    composition_list = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        string = string.replace(letter, ","+letter)
    string = string.lstrip(',')
    return string.split(',')

def get_fraction(composition):
    """Given a string like Al22.2, get the fraction i.e. 22.2
    """
    numerical_value = "".join([i for i in composition[1:] if (i.isnumeric() or i=='.')])
    return float(numerical_value) if numerical_value!="" else 1.0

def extract_elem_from_string(composition):
    """ Given a string like Al22.2, get the element symbol, i.e. Al
    """
    return "".join([i for i in composition[:2] if i.isalpha()])

############################## Functions to find matching physical properties about thte element #######
def get_natural_average_atomic_mass(elem, verbose=False, **kwargs):
    """ returns the average atomic mass of the element when it is in its naturally occurring ratio of isotopic abundances.
    **kwargs is plugged into the numpy.isclose function ot modify rtol and atol.
    """
    isotopic_abundance = {isotope: abundance for isotope, abundance in NATURAL_ABUNDANCE.items() if extract_elem_from_string(isotope)==elem}
    if sum(isotopic_abundance.values())!=1.0:
        if verbose:
            print("Expected the isotopic abundances of {} to sum to 1.0, got {} instead.".format(elem, sum(isotopic_abundance.values())))
        if not np.isclose(sum(isotopic_abundance.values()), 1.0, **kwargs):
            raise ValueError("The sum of abundances for element '{}' = {} is not close to unity.".format(elem, sum(isotopic_abundance.values())))
    return sum([atomic_mass(isotope)*abundance for isotope, abundance in isotopic_abundance.items()])

def get_elemental_fractions(formula):
    if ("/" not in formula) or ("(Atomic%)" in formula):
        # use atomic percentage
        compositions = split_at_cap(formula.replace("(Atomic%)", "")) # could've used removesuffix in python 3.9; but I've not installed that yet.
        elements = [extract_elem_from_string(comp) for comp in compositions]

        raw_mole_ratios = [get_fraction(comp) for comp in compositions] # interpret them as mole ratios

        elemental_fractions = {elem:mole_ratio/sum(raw_mole_ratios) for mole_ratio, elem in zip(raw_mole_ratios, elements)}
    else:
        # is an alloy, use wt. percentage
        compositions = expand_bracket(formula).split("/")
        elements = [extract_elem_from_string(comp) for comp in compositions]

        weight_fractions = [get_fraction(comp)/100.0 for comp in compositions] # interpret them as weight fraction
        average_mass = 1/sum([weight/get_natural_average_atomic_mass(elem) for weight, elem in zip(weight_fractions, elements)]) # obtain the average mass which is then used as a multiplier below:

        elemental_fractions = { elem: weight/get_natural_average_atomic_mass(elem)*average_mass for weight, elem in zip(weight_fractions, elements) }
    return elemental_fractions

def convert_elemental_to_isotopic_fractions(dict_of_elemental_fractions):
    isotopic_fractions = {}
    for elem, fraction in dict_of_elemental_fractions.items():
        matching_isotopes = [iso for iso in NATURAL_ABUNDANCE.keys() if extract_elem_from_string(iso)==elem]
        for isotope in matching_isotopes:
            isotopic_fractions[isotope] = NATURAL_ABUNDANCE[isotope]*fraction
    return isotopic_fractions

def get_average_atomic_mass_from_isotopic_fractions(dict_of_isotopic_fractions):
    return sum([atomic_mass(isotope)*fraction for isotope, fraction in dict_of_isotopic_fractions.items()])

def get_average_atomic_mass_from_elemental_fractions(dict_of_elemental_fractions):
    return sum([get_natural_average_atomic_mass(elem)*fraction for elem, fraction in dict_of_elemental_fractions.items()])

def pick_material(species, expanded_physical_prop_df):
    """Choose the material with the highest atomic fraction of this species
    """
    material = expanded_physical_prop_df[species].idxmax()
    if expanded_physical_prop_df[species][material]>0:
        return expanded_physical_prop_df.loc[material]
    else:
        dummy_data_series = pd.Series({col: float('nan') for col in expanded_physical_prop_df.columns})
        dummy_data_series['Formula'] = 'N/A'
        dummy_data_series.name = 'Missing'
        return dummy_data_series

def pick_all_usable_materials(species, expanded_physical_prop_df):
    """Pick all materials that has even a trace of the species of interest.
    """
    material = expanded_physical_prop_df[species]>0
    return expanded_physical_prop_df.loc[material]