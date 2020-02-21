from numpy import exp, cos, arccos, sin, arctan, tan, pi, sqrt; import numpy as np; tau = 2*pi
from numpy import array as ary
from numpy import log as ln
import pandas as pd
import json
from matplotlib import pyplot as plt
import uncertainties
from openmc.data import NATURAL_ABUNDANCE
import os, sys
from convert2R import HPGe_efficiency_curve_generator, load_R_rr_radiation, USE_NATURAL_ABUNDANCE, THRESHOLD_ENERGY, EVALUATE_THERMAL, serialize
from collapx import flux_conversion, MeV
from openmc.data import atomic_weight, atomic_mass
import copy

MIN_MELTING_POINT = 20 # Degree Celcius
VERBOSE = False 
SATURATION_COUNT_RATE = 10E-2/4E-6 #s^-1 # 10% dead time, assuming each pulse lasting a maximum of 4 microseconds
AREA = 1 # cm^2 # The bigger the area, the less thickness is required to reach a detectible limit, thus self-shielding distortion of the true spectrum.
# AMPLIFY_LOW_E_GAMMA = False # fix the detection efficiency = 0.25 for gamma with energy lower than THRESHOLD_ENERGY[0].
#The parameter below is for debugging use only.
CM_MM_CONVERSION = 10 #
#The only disadvantage to having a foil with too much area is that it must be very thin,
#perhaps impossible-to-manufature-ly thin, not overexpose the detector.

def turn_str_back_into_Var(data):
    if isinstance(data, str):
        try:
            values = data.split("+/-")
            data = uncertainties.core.Variable(*[float(i) for i in values])
        except:
            pass
    elif isinstance(data, dict):
        for k,v in data.items():
            data[k] = turn_str_back_into_Var(v)
    elif isinstance(data, list): #then it is a class
        data = [turn_str_back_into_Var(i) for i in data]
    else:
        pass
    return data

def insert_at_cap(string, char='/'):
    new_string = [string[0]]
    for i in range(1,len(string)):
        c = string[i]
        if c.isupper() and string[i-1]!='/':
            new_string.append(char)
        new_string.append(c)
    return "".join(new_string)

def but_actually_in(word, lst_of_words): 
    new_list = [] 
    for string in lst_of_words: 
        if word in [ remove_not_alpha(i) for i in insert_at_cap(string).split('/') ]: 
            new_list.append(string) 
    return new_list

phyprop_fname = 'physicalprop.csv'
try:
    # assert not sys.argv[-1].endswith('.py'), "An extra argument should be parsed, indicating the output directory"
    physical_properties = pd.read_csv(os.path.join(sys.argv[-1], phyprop_fname))
except FileNotFoundError:
    print(f"No files named {phyprop_fname} found in {sys.argv[-1]}, defaulting to using file {'goodfellow_'+phyprop_fname} in the same directory as {sys.argv[0]}")
    physical_properties = pd.read_csv('goodfellow_physicalprop.csv', index_col=1, na_values='-')

remove_not_alpha = lambda p: "".join([c for c in p if c.isalpha()])

def scrape_all_non_alpha_numeric(string):
    new_string = []
    for c in string:
        if c.lower() in "1234567890qwertyuiopasdfghjklzxcvbnm./":
            new_string.append(c)
    return "".join(new_string)

def return_matching(list_of_strings, element):
    for comp in list_of_strings:
        if element in comp:
            return comp

def find_closest_match(list_of_solids, element):
    composiitions = [scrape_all_non_alpha_numeric(insert_at_cap(formula).replace(")","/")).split('/') for formula in list_of_solids]
    elem = [ return_matching(comp, element) for comp in composiitions]
    excess = [e[len(element):] for e in elem]
    frac = []
    #follows the order of
    #Cc99.n
    #Cc9n
    #Ccnn
    #Ccn/Cc
    for e in excess:
        try:
            frac.append(float(e))
        except ValueError:
            #Any other matches
            frac.append(1.0)
    return np.argmax(frac)

def get_physical_prop(element):
    if element in list(physical_properties.index):
        return physical_properties.loc[element]
    else:
        close_match = but_actually_in(element, list(physical_properties.index))
        if len(close_match)==0:
            return None
        if VERBOSE:
            print("Choices for", element, "include", close_match)
        best_guess = close_match[find_closest_match(close_match, element)]
        print(f"No exact match for {element}, use closest match:{best_guess} manual correction for the dilution may be needed.")
        return physical_properties.loc[best_guess]

material_name, density, melting_point = list(physical_properties.columns)
def get_density(props):
    if props is not None:
        if "Nickel/Silicon"== props[material_name]: 
            return 2.329
        if np.isnan(props[density]):
            print(f"Ignoring {props[material_name]} ({props.name}) as its density value is not found. ")
        else:
            return props[density]
    else:
        print(f"Ignoring {props[material_name]} ({props.name}) as no matching solid is found.")
        return None

def compare_mp(props, mp):
    if props is not None:
        if np.isnan(props[melting_point]):
            print("ignoring t") # for future compatibility
            print("No melting point found for {0}, accepting it by default.".format(props[material_name]))
            return True
        elif props[melting_point]>=mp:
            return True
        else:
            if VERBOSE:
                print(f"rejecting {element} which has melting point {prop[melting_point]} < {mp}.")
            return False
    else:
        print("No solid found for {0}, accepting it by default".format(element))
        return True

def get_all_parents(rname_list):
    parents = [] # either contains the isotopes or the elements
    if USE_NATURAL_ABUNDANCE:
        for r in rname_list:
            e = ''.join([c for c in r.split('-')[0] if c.isalpha()])
            if e not in parents:
                parents.append(e)
        print(len(parents), "elements are considered.\n")
    else:
        for r in rname_list:
            e = r.split('-')[0]
            if e not in parents:
                parents.append(e)
        print(len(parents), "isotopes are considered.\n")
    return parents

def check_if_match(long_name, short_name):
    s = len(short_name)
    if long_name[:s]!=short_name: #long_name-short_name !=0
        return False
    elif len(long_name)==s: # no more characters left on the long name
        return True
    else:
        if long_name[s].isalpha(): # one more letter left
            return False
        else:
            return True

def add_saturation_thicknesses(parents, spectra_json):
    total_counts_pmp = []
    saturation_thicknesses = {}
    for p in parents:
        count_rate_pmp = []
        usable_counts_pmp_of, thermal_contribution = {}, {}
        for rname, products in spectra_json.items():
            if rname.startswith(p):
                usable_counts_pmp_of[rname] = []
                thermal_contribution[rname] = {}
                for prod_name, prod in products.items():
                    thermal_contribution[rname] = { prod_name: [] }
                    N0_pmp = prod['measurement']['N_0']
                    dec_constant = prod['measurement']['decay_constant']
                    dec_corr_fac = prod['measurement']['decay_correct_factor']
                    meas_time = prod['measurement']['measurement_time']
                    irr_period = prod['measurement']['irradiation_time']
                    for rad_name, rad in prod.items():
                        if rad_name in ['gamma', 'xray']:
                            for radiation in rad['discrete']:
                                if False:
                                    # if radiation['energy']<THRESHOLD_ENERGY[0] and AMPLIFY_LOW_E_GAMMA:
                                    number_reaching_detector = (N0_pmp  * radiation['intensity']/100 *0.25).n# keep the photopeak efficiency at 50% when the energy is less than 100keV
                                    count_rate_pmp.append() 
                                else:
                                    number_reaching_detector = (N0_pmp * radiation['intensity']/100 * radiation['efficiency'] ).n
                                    radiation['counts per mole parent'] = number_reaching_detector
                                count_rate_pmp.append(number_reaching_detector * dec_constant.n)
                                if THRESHOLD_ENERGY[0]<=radiation['energy']<=THRESHOLD_ENERGY[1]:
                                    usable_counts_pmp_of[rname].append( number_reaching_detector * (1-exp(-dec_constant.n*meas_time)) )
                                # I am aware that the efficiency curve is not accurate when extrapolated to lower energies.
                                # However we can use various tricks with the electronics to isolate the lower energy pulses.
                                # so it shouldn't matter too much.
                                ### NEED TO CONFIRM THIS using AMPLIFY_LOW_E_GAMMA
                                contr_counts = prod['measurement']['area_pmp'] * dec_corr_fac * (1-exp(-dec_constant.n*meas_time)) * irr_period * radiation['intensity']/100 * radiation['efficiency']
                                thermal_contribution[rname][prod_name].append(contr_counts)
        total_count_rate_pmp = sum(count_rate_pmp)
        max_mole_parent = SATURATION_COUNT_RATE / total_count_rate_pmp
        if USE_NATURAL_ABUNDANCE:
            props = get_physical_prop(p)
        else:
            props = get_physical_prop(remove_not_alpha(p[:2]))
        if props is not None:
            if (dense:=get_density(props)) is not None and compare_mp(props, MIN_MELTING_POINT): 
                if USE_NATURAL_ABUNDANCE:
                    vol_pm = atomic_weight(p)/dense
                else:
                    vol_pm = atomic_mass(p)/dense
                
                prop_dict = dict(props.copy())
                prop_dict.update({"volume per mole (cm3)":vol_pm, 'formula': props.name})
                
                max_vol = max_mole_parent*vol_pm
                max_thickness = max_vol/AREA
                saturation_thicknesses[p] = {
                "maximum thickness (cm)":max_thickness,
                "total count rate per mole parent":total_count_rate_pmp,
                "physical properties": prop_dict,
                "usable counts per unit thickness": {rname: ary(counts)/vol_pm*AREA for rname, counts in usable_counts_pmp_of.items()},
                # "counts/(thickness*flux)": {rname: np.sum(list(contr.values()), axis=0)/vol_pm*AREA for rname, contr in thermal_contribution.items()}
                }
        else:
            print("No matching entries found for", p)
    return saturation_thicknesses

def add_min_thickness(R, rr, spectra_json, saturation_thicknesses):
    thickness_range_df, rname_list = [], []
    for reaction_name, reaction_info in rr.iterrows():
        # Nk_pmp = reaction_info['N_infty per mole parent']
        min_mol_parent = reaction_info['min mole of parent']
        for p, info in saturation_thicknesses.items():
            is_matching = check_if_match(insert_at_cap(reaction_name, '-').split('-')[0], p)
            if is_matching:
                vol_pm = info['physical properties']['volume per mole (cm3)']
                min_vol = min_mol_parent * vol_pm
                min_thickness = min_vol/AREA
                max_thickness = info['maximum thickness (cm)']
                
                mole_per_cm_thick = AREA/info['physical properties']['volume per mole (cm3)']
                
                count_rate_per_vol = info['total count rate per mole parent'] / vol_pm
                count_rate_per_thickness = count_rate_per_vol * AREA
                material = info['physical properties'][material_name]
                formula = info['physical properties']['formula']
                half_lives = [prod['measurement']['half_life'] for prod in spectra_json[reaction_name].values()]
                if min_thickness>max_thickness:
                    print(reaction_name, "should be discarded as a larger volume than saturation count rate is required to have an appreciable effectiveness when unfolding.")
                thickness_range_df.append([
                    min_thickness,
                    max_thickness,
                    mole_per_cm_thick,
                    count_rate_per_thickness,
                    material,
                    formula,
                    info['physical properties'][density],
                    half_lives,
                    sum(info["usable counts per unit thickness"][reaction_name]),
                    #info["counts/(thickness*flux)"][reaction_name]])
                    ])
                rname_list.append(reaction_name)
    
    return pd.DataFrame(thickness_range_df, columns=[
        'min thickness',
        'max thickness',
        'mole per cm thickness',
        'total count rate per cm thickness',
        'material',
        'formula',
        'density value used',
        'half-lives',
        'sum of usable counts from all peaks',
        # 'counts per cm thickness per cm-2s-1 thermal flux',
        ], index=rname_list)

if __name__=="__main__":
    R, rr, spectra_json = load_R_rr_radiation(sys.argv[-1])
    turn_str_back_into_Var(spectra_json)
    parents = get_all_parents(rr.index)
    saturation_thicknesses = add_saturation_thicknesses(parents, spectra_json)
    # overwrite the old version, so that the 'counts per mole parent' is added
    spectra_file = os.path.join(sys.argv[-1], "spectra.json")
    spec_copy = copy.deepcopy(spectra_json)
    with open(spectra_file, mode='w', encoding='utf-8') as f:
        json.dump(serialize(spec_copy), f)
    
    #save the thickness_range_df
    thickness_range_df = add_min_thickness(R, rr, spectra_json, saturation_thicknesses)
    thickness_range_df.to_csv(os.path.join(sys.argv[-1], "thicknesses.csv"), index_label='rname')