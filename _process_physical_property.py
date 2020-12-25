from numpy import array as ary; import numpy as np
import pandas as pd
import os, sys, itertools
from scipy.constants import Avogadro
from misc_library import (  extract_elem_from_string,
                            get_natural_average_atomic_mass,
                            get_elemental_fractions,
                            convert_elemental_to_isotopic_fractions,
                            get_average_atomic_mass_from_isotopic_fractions,
                            get_average_atomic_mass_from_elemental_fractions)
from misc_library import ATOMIC_NUMBER, NATURAL_ABUNDANCE, atomic_mass, REACTION_NAME, SUM_RULES
"""
Module to algorithmically generate information about the the number densities and atomic fractions of solids, given a prompt csv that contains only 
    densities
    chemcial formula
    material name
    (Melting point)
of the material.
"""

ADD_ELEMENTAL_FRACTION = True
ADD_ISOTOPIC_FRACTION = True
PHYSICAL_PROP_FILE = "goodfellow_physicalprop.csv"
if __name__=='__main__':

    print(f"Reading ./{os.path.relpath(PHYSICAL_PROP_FILE)} to be interpreted as the physical parameters")
    physical_prop = pd.read_csv(PHYSICAL_PROP_FILE, index_col=[0], na_values='-', comment='#')
    elemental_fractions, isotopic_fractions, average_atomic_mass = {}, {}, {}

    for material, _property in physical_prop.iterrows():
        elemental_fractions[material] = get_elemental_fractions(_property.Formula)
        isotopic_fractions[material] = convert_elemental_to_isotopic_fractions(elemental_fractions[material])
        average_atomic_mass[material] = get_average_atomic_mass_from_isotopic_fractions(isotopic_fractions[material])

    physical_prop['Average atomic mass'] = pd.Series(average_atomic_mass)

    physical_prop['Number density-cm-3'] = Avogadro/physical_prop['Average atomic mass'] * physical_prop['Density-gcm-3']

    all_isotopes = [i[3:] for i in sorted( # sort it, and then remove the extraneous atomic numbers
                        [str(ATOMIC_NUMBER[extract_elem_from_string(iso)]).zfill(3)+iso # add the atomic number in front of each
                            for iso in set(
                                itertools.chain.from_iterable(
                                    [list(d.keys()) for d in isotopic_fractions.values()]) # get all of the isotope names
                                    )])]

    # add the current dictionary of isotopes into the dataframe
    if ADD_ISOTOPIC_FRACTION:
        for isotope_type in all_isotopes:
            fraction_of_this_isotope = {}
            for material in physical_prop.index:
                if isotope_type in isotopic_fractions[material].keys():
                    fraction_of_this_isotope[material] = isotopic_fractions[material][isotope_type]
                else:
                    fraction_of_this_isotope[material] = 0.0
            physical_prop[isotope_type] = pd.Series(fraction_of_this_isotope)
        check_fraction_sum = physical_prop[all_isotopes].sum(axis=1)
        for material, sum_value in check_fraction_sum.items():
            assert np.isclose(sum_value, 1.0), "Expected the isotopic fraction of {} to sum to around 1.0; got {} instead.".formta(material, sum_value)

    # add the current dictionary of elements into the dataframe
    if ADD_ELEMENTAL_FRACTION:
        all_elements = [i for i in ATOMIC_NUMBER.keys() if i!='n']
        for elem in all_elements:
            fraction_of_this_element = {}
            for material in physical_prop.index:
                if elem in elemental_fractions[material].keys():
                    fraction_of_this_element[material] = elemental_fractions[material][elem]
                else:
                    fraction_of_this_element[material] = 0.0
            physical_prop[elem] = pd.Series(fraction_of_this_element)

        check_fraction_sum = physical_prop[all_elements].sum(axis=1)
        for material, sum_value in check_fraction_sum.items():
            assert np.isclose(sum_value, 1.0), "Expected the elemental fraction of {} to sum to around 1.0; got {} instead.".formta(material, sum_value)

    filename = 'physical_property.csv'
    if ADD_ISOTOPIC_FRACTION:
        filename = 'isotopic_frac_'+filename
    if ADD_ELEMENTAL_FRACTION:
        filename = 'elemental_frac_'+filename
    physical_prop.to_csv(filename)