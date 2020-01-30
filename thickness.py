from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; import numpy as np; tau = 2*pi
from numpy import array as ary
from numpy import log as ln
import pandas as pd
import json
from matplotlib import pyplot as plt
import uncertainties
from openmc.data import NATURAL_ABUNDANCE
import os, sys
from convert2R import HPGe_efficiency_curve_generator, load_rr_R_radiation, USE_NATURAL_ABUNDANCE, THRESHOLD_ENERGY, IRRADIATION_DURATION
from openmc.data import atomic_weight, atomic_mass

MIN_MELTING_POINT = 20 # Degree celcius
AMPLIFY_LOW_E_GAMMA = False
SATURATION_COUNT_RATE = 10E-2/4E-6 #s^-1 # 10% dead time, with each pulse lasting a maximum of 4 microseconds
VERBOSE = False
AREA = pi*3.0**2 # The bigger the area, the less thickness is required to reach a detectible limit, thus self-shielding distortion of the true spectrum.
CM_TO_MM = 10 #
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
        print("No exact match, use closest match:", best_guess, ", manual correction for the dilution may be needed.")
        return physical_properties.loc[best_guess]

material_name, density, melting_point = list(physical_properties.columns)
def get_density(element):
    props = get_physical_prop(element)
    if props is not None:
        if np.isnan(props[density]):
            print("Ignoring 1 entry as the density value for ",element,"is not found")
        else:
            return props[density]
    else:
        print("Ignoring 1 entry as no matching solid is found for", element)
        return None

if __name__=="__main__":
    R, rr, spectra_json = load_rr_R_radiation(sys.argv[-1])
    turn_str_back_into_Var(spectra_json)

    parents = [] # either contains the isotopes or the elements
    if USE_NATURAL_ABUNDANCE:
        for r in R.rname:
            e = ''.join([c for c in r.split('-')[0] if c.isalpha()])
            if e not in parents:
                parents.append(e)
        print(len(parents), "elements are considered.\n")
    else:
        for r in R.rname:
            e = r.split('-')[0]
            if e not in parents:
                parents.append(e)
        print(len(parents), "isotopes are considered.\n")
    total_counts_pmp = []
    for p in parents:
        count_rate_pmp = []
        for iso, products in spectra_json.items():
            if iso.startswith(p):
                for prod in products.values():
                    N0_pmp = prod['measurement']['N_0']
                    for rad_name, rad in prod.items():
                        if rad_name in ['gamma', 'xray']:
                            for radiation in rad['discrete']:
                                if radiation['energy']<THRESHOLD_ENERGY[0] and AMPLIFY_LOW_E_GAMMA:
                                    count_rate_pmp.append(N0_pmp * radiation['intensity']/100 *0.25) # keep the photopeak efficiency at 50% when the energy is less than 100keV
                                else:
                                    count_rate_pmp.append(N0_pmp * radiation['intensity']/100 * radiation['efficiency'] )
                                # I am aware that the efficiency curve is not accurate when extrapolated to lower energies.
                                # However we can use various tricks with the electronics to isolate the lower energy pulses.
                                # so it shouldn't matter too much.
                                ### NEED TO CONFIRM THIS using AMPLIFY_LOW_E_GAMMA
        total_counts_pmp.append(sum(count_rate_pmp))
        max_mole_of_parent = SATURATION_COUNT_RATE/sum(count_rate_pmp)
        if USE_NATURAL_ABUNDANCE:
            dense = get_density(p)
        else:
            dense = get_density(remove_not_alpha(p[:2]))
        if dense is not None:
            vol_pm = atomic_weight(p)/dense
        max_vol=max_mole_of_parent*vol_pm
        thickness = max_vol/AREA
        print(thickness*CM_TO_MM, "mm of foil is allowed for", p)
        # print(p, sum([i.n for i in count_rate_pmp]))