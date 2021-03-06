# typical system/python stuff
import os, sys, json
import warnings, gc
from io import StringIO
from collections import OrderedDict
from tqdm import tqdm
# typical python numerical stuff
from numpy import array as ary; import numpy as np
from numpy import log as ln
import pandas as pd
# from matplotlib import pyplot as plt
# openmc stuff
import openmc
from openmc.data import IncidentNeutron, Decay, Evaluation, ATOMIC_SYMBOL
from openmc.data.reaction import REACTION_NAME
# uncertainties
import uncertainties
from uncertainties.core import Variable
# custom modules
from flux_convert import Integrate
from misc_library import (haskey,
                         # plot_tab,
                         tabulate,
                         detabulate,
                         ordered_set,
                         EncoderOpenMC,
                         MT_to_nuc_num, 
                         load_endf_directories,
                         FISSION_MTS, AMBIGUOUS_MT,
                         HPGe_efficiency_curve_generator,
                         save_parameters_as_json)

HPGe_eff_file = '.physical_parameters/Absolute_photopeak_efficiencyMeV.csv'
photopeak_eff_curve = HPGe_efficiency_curve_generator(HPGe_eff_file)
gamma_E = [20*1E3, 4.6*1E6] # detectable gamma energy range

def sort_and_trim_ordered_dict(ordered_dict, trim_length=3):
    """
    sort an ordered dict AND erase the first three characters (the atomic number) of each name
    """
    return OrderedDict([(key[trim_length:], val)for key,val in sorted(ordered_dict.items())])

def extract_decay(dec_file):
    """
    extract the useful information out of an openmc.data.Decay entry
    """
    half_life = Variable(np.clip(dec_file.half_life.n, 1E-23, None), dec_file.half_life.s) # increase the half-life to a non-zero value.
    decay_constant = ln(2)/(half_life)
    modes = {}
    for mode in dec_file.modes:
        modes[mode.daughter] = mode.branching_ratio
    return dict(decay_constant=decay_constant, branching_ratio=modes, spectra=dec_file.spectra)

def condense_spectrum(dec_file):
    """
    We will explicitly ignore all continuous distributions because they do not show up as clean gamma lines.
    Note that the ENDF-B/decay/ directory stores a lot of spectra (even those with very clear lines) as continuous,
        so it may lead to the following method ignoring it.
        The workaround is just to use another library where the evaluators aren't so lazy to not use the continuous_flag=='both' :/
    """
    count = Variable(0.0,0.0)
    if ("gamma" in dec_file['spectra']) and ("discrete" in dec_file['spectra']['gamma']):
        norm_factor = dec_file['spectra']['gamma']['discrete_normalization']
        for gamma_line in dec_file['spectra']['gamma']['discrete']:
            if np.clip(gamma_line['energy'].n, *gamma_E)==gamma_line['energy'].n:
                count += photopeak_eff_curve(gamma_line['energy']) * gamma_line['intensity']/100 * norm_factor
    if ('xray' in dec_file['spectra']) and ('discrete' in dec_file['spectra']['xray']):
        norm_factor = dec_file['spectra']['xray']['discrete_normalization']
        for xray_line in dec_file['spectra']['xray']['discrete']:
            if np.clip(xray_line['energy'].n, *gamma_E)==xray_line['energy'].n:
                additional_counts = photopeak_eff_curve(xray_line['energy']) * xray_line['intensity']/100 * norm_factor
                if not additional_counts.s<=additional_counts.n:
                    additional_counts = Variable(additional_counts.n, additional_counts.n) # clipping the uncertainty so that std never exceed the mean. This takes care of the nan's too.
                count += additional_counts
    del dec_file['spectra']
    dec_file['countable_photons'] = count #countable photons per decay of this isotope
    return

def deduce_daughter_from_mt(parent_atomic_number, parent_atomic_mass, mt):
    """
    Given the atomic number and mass number, get the daughter in the format of 'Ag109'.
    """
    if mt in MT_to_nuc_num.keys():
        element_symbol = openmc.data.ATOMIC_SYMBOL[parent_atomic_number + MT_to_nuc_num[mt][0]]
        product_mass = str(parent_atomic_mass + MT_to_nuc_num[mt][1])
        if len(MT_to_nuc_num[mt])>2 and MT_to_nuc_num[mt][2]>0: # if it indicates an excited state
            excited_state = '_e'+str(MT_to_nuc_num[mt][2])
            return element_symbol+product_mass+excited_state
        else:
            return element_symbol+product_mass
    else:
        return None

class Tab1DWithAddition(openmc.data.Tabulated1D):
    def __add__(self, tab_like):
        pass
        return tab_like

def extract_xs(parent_atomic_number, parent_atomic_mass, rx_file, tabulated=True):
    """
    For a given (mf, mt) file,
    Extract only the important bits of the informations:
    actaul cross-section, and the yield for each product.
    Outputs a list of these modified cross-sections (which are multiplied onto the thing if possible)
        along with all their names.
    The list can then be added into the final reaction dictionary one by one.
    """
    appending_name_list, xs_list = [], []
    xs = rx_file.xs['0K']
    if isinstance(xs, openmc.data.ResonancesWithBackground):
        xs = xs.background # When shrinking the group structure, this contains everything you need. The Resonance part of xs can be ignored (only matters for self-shielding.)
    if len(rx_file.products)==0: # if no products are already available, then we can only assume there is only one product.
        daughter_name = deduce_daughter_from_mt(parent_atomic_number, parent_atomic_mass, rx_file.mt)
        if daughter_name: # if the name is not None or False, i.e. a matching MT number is found.
            name = daughter_name+'-MT='+str(rx_file.mt) # deduce_daughter_from_mt will return the ground state value
            appending_name_list.append(name)
            xs_list.append(detabulate(xs) if (not tabulated) else xs)
    else:
        for prod in rx_file.products:
            appending_name_list.append(prod.particle+'-MT='+str(rx_file.mt))
            partial_xs = openmc.data.Tabulated1D(xs.x, prod.yield_(xs.x) * xs.y,
                                                breakpoints=xs.breakpoints, interpolation=xs.interpolation)
            xs_list.append(detabulate(partial_xs) if (not tabulated) else partial_xs)
    return appending_name_list, xs_list

def collapse_xs(xs_dict, gs_ary):
    """
    Calculates the group-wise cross-section in the given group-structure
    by averaging the cross-section within each bin.
    """
    collapsed_sigma = dict()
    print("Collapsing the cross-sections to the desired group-structure:")
    for parent_product_mt, xs in tqdm(xs_dict.items()):
        # perform the integration
        I = Integrate(xs)
        sigma = I.definite_integral(*gs_ary.T)/np.diff(gs_ary, axis=1).flatten()

        collapsed_sigma[parent_product_mt] = sigma
    del xs_dict; gc.collect()
    return pd.DataFrame(collapsed_sigma).T

def merge_identical_parent_products(sigma_df):
    # get the parent_product string and mt number string as two list, corresponding to each row in the sigma_df.
    parent_product_list, mt_list = [], []
    for parent_product_mt in sigma_df.index:
        parent_product_list.append("-".join(parent_product_mt.split("-")[:2]))
        mt_list.append(parent_product_mt.split("=")[1])
    parent_product_list, mt_list = ary(parent_product_list), ary(mt_list) # make them into array to make them indexible.

    partial_reaction_array = sigma_df.values
    parent_product_all = ordered_set(parent_product_list)

    sigma_unique = {}
    print("\nCondensing the sigma_df dataframe to merge together reactions with identical (parent, product) pairs:")
    for parent_product in tqdm(parent_product_all):
        matching_reactions = parent_product_list==parent_product
        mt_name = "-MT=({})".format(",".join(mt_list[matching_reactions]))
        sigma_unique[parent_product+mt_name] = partial_reaction_array[matching_reactions].sum(axis=0)
    del sigma_df; del partial_reaction_array; gc.collect()
    return pd.DataFrame(sigma_unique).T

class MF10(object):
    def __getitem__(self, key):
        return self.reactions.__getitem__(key)

    def __len__(self):
        return self.reactions.__len__()

    def __iter__(self):
        return self.reactions.__iter__()

    def __reversed__(self):
        return self.reactions.__reversed__()

    def __contains__(self, key):
        return self.reactions.__contains__(key)

    __slots__ = ["number_of_reactions", "za", "target_mass", "target_isomeric_state", "reaction_mass_difference", "reaction_q_value", "reactions"]
    # __slots__ created for memory management purpose in case there are many suriving instances of MF10 all present at once.
    def __init__(self, mf10_mt5_section):
        if mf10_mt5_section is not None:
            file_stream = StringIO(mf10_mt5_section)
            za, target_mass, target_iso, _, ns, _ = openmc.data.get_head_record( file_stream ) # read the first line, i.e. the head record for MF=10, MT=5.
            self.number_of_reactions = ns
            self.za = za
            self.target_mass = target_mass
            self.target_isomeric_state = target_iso
            self.reaction_mass_difference, self.reaction_q_value = {}, {}
            self.reactions = {}
            for reaction_number in range(ns):
                (mass_diff, q_value, izap, isomeric_state), tab = openmc.data.get_tab1_record(file_stream)
                self.reaction_mass_difference[(izap, isomeric_state)] = mass_diff
                self.reaction_q_value[(izap, isomeric_state)] = q_value
                self.reactions[(izap, isomeric_state)] = tab
        else:
            self.reactions = {}

    def keys(self):
        return self.reactions.keys()

    def items(self):
        return self.reactions.items()

    def values(self):
        return self.reactions.values()

def merge_xs(low_E_xs, high_E_xs, debug_info=""):
    """
    Specifically created to append E>=30MeV xs onto the normally obtained data.
    Adds together two different cross-section profiles.
    The high_E_xs is expected to start at a minimum energy E_min,
    while low_E_xs is expected to have its xs zeroed at higher than xs.

    Otherwise... well fuck.
    I'm going to have to add them together and figure out some sort of interpoation scheme.
    """
    E_min = high_E_xs.x.min()
    low_E_range = low_E_xs.x<E_min
    strictly_high_E_range = low_E_xs.x>E_min # ignore the point E = E_min
    # assert all(low_E_xs.y[low_E_range]==0.0), "The high_E_xs record must start at E_min, and the low_E_xs record is expected to have zeroed all y values above at x>=E_min."
    if all(low_E_xs.y[strictly_high_E_range]==0.0):
        low_x, low_y, low_interp = low_E_xs.x[low_E_range], low_E_xs.y[low_E_range], ary(detabulate(low_E_xs)["interpolation"], dtype=int)[low_E_range[:-1]]
        high_x, high_y, high_interp = high_E_xs.x, high_E_xs.y, ary(detabulate(high_E_xs)["interpolation"], dtype=int)
        return tabulate({
            "x":np.hstack([low_x, high_x]),
            "y":np.hstack([low_y, high_y]),
            "interpolation":np.hstack([low_interp, high_interp]),
            })
    else:
        plot_tab(low_E_xs)
        plot_tab(high_E_xs)
        plt.title(debug_info)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()
        return low_E_xs

# must load decay informatino and neutron incidence information in a single python script,
# because the former is needed to compile the dictionary of isomeric_to_excited_state, which the latter use when saving the reaction rates products..
if __name__=='__main__':
    assert os.path.exists(os.path.join(sys.argv[-1], 'gs.csv')), "Output directory must already have gs.csv"
    gs = pd.read_csv(os.path.join(sys.argv[-1], 'gs.csv')).values
    if (SORT_BY_REACTION_RATE:=True):
        assert os.path.exists(os.path.join(sys.argv[-1], 'integrated_apriori.csv')), "Output directory must already have integrated_apriori.csv in order to sort the response.csv in descending order of expected-radionuclide-population later on."
        apriori = pd.read_csv(os.path.join(sys.argv[-1], 'integrated_apriori.csv'))['value'].values
    endf_file_list = load_endf_directories(sys.argv[1:])
    print(f"Loaded {len(endf_file_list)} different material files,\n")

    # First compile the decay records
    print("\nCompiling the decay information as decay_dict, and recording the excited-state to isomeric-state information:")
    decay_dict = OrderedDict() # dictionary of decay data
    isomeric_to_excited_state = OrderedDict() # a dictionary that translates all 
    with warnings.catch_warnings(record=True) as w_list:
        for file in tqdm(endf_file_list):
            name = str(file.target["atomic_number"]).zfill(3) + ATOMIC_SYMBOL[file.target["atomic_number"]] + str(file.target["mass_number"])
            isomeric_name = name # make a copy
            if file.target["isomeric_state"]>0: # if it is not at the lowest isomeric state: add the _e behind it too.
                isomeric_name += "_m"+str(file.target["isomeric_state"])
                name += "_e"+str(file.target["state"])
            isomeric_to_excited_state[isomeric_name] = name[3:] # trim the excited state name

            if file.info['sublibrary']=="Radioactive decay data": # applicable to materials with (mf, mt) = (8, 457) file section
                dec_f = Decay.from_endf(file)
                decay_dict[name] = extract_decay(dec_f)
    if w_list:
        print(w_list[0].filename+", line {}, {}'s:".format(w_list[0].lineno, w_list[0].category.__name__))
        for w in w_list:
            print("    "+str(w.message))
    decay_dict = sort_and_trim_ordered_dict(decay_dict) # reorder it so that it conforms to the 
    isomeric_to_excited_state = sort_and_trim_ordered_dict(isomeric_to_excited_state)
    
    # Save said decay records
    with open(os.path.join(sys.argv[-1], FULL_DECAY_INFO_FILE:='decay_radiation.json'), 'w') as f:
        print("\nSaving the decay spectra as {} ...".format(FULL_DECAY_INFO_FILE))
        json.dump(decay_dict, f, cls=EncoderOpenMC)

    # turn decay records into number of counts
    print("\nCondensing each decay spectrum...")
    for name, dec_file in tqdm(decay_dict.items()):
        condense_spectrum(dec_file)

    with open(os.path.join(sys.argv[-1], CONDENSED_DECAY_INFO_FILE:='decay_counts.json'), 'w') as f:
        print("\nSaving the condensed decay information as {} ...".format(CONDENSED_DECAY_INFO_FILE))
        json.dump(decay_dict, f, cls=EncoderOpenMC)
        
    # Then compile the Incident-neutron records
    print("\nCompiling the raw cross-section dictionary.")
    xs_dict = OrderedDict()
    for file in tqdm(endf_file_list):
        # mf10 = {}
        if file.info['sublibrary']=="Incident-neutron data":
            inc_f = IncidentNeutron.from_endf(file)
            nuc_sort_name = str(inc_f.atomic_number).zfill(3)+inc_f.name

            # get the higher-energy range values of xs as well if available.
            mf10_mt5 = MF10(file.section.get((10, 5), None)) # default value = None if (10, 5 doesn't exist.)
            for (izap, isomeric_state), xs in mf10_mt5.items():
                atomic_number, mass_number = divmod(izap, 1000)
                if atomic_number>0 and mass_number>0: # ignore the weird products that means nothing meaningful
                    isomeric_name = ATOMIC_SYMBOL[atomic_number]+str(mass_number)
                    if isomeric_state>0: 
                        isomeric_name += "_m"+str(isomeric_state)
                    e_name = isomeric_to_excited_state.get(isomeric_name, isomeric_name.split("_")[0]) # return the ground state name if there is no corresponding excited state name for such isomer.
                    long_name = nuc_sort_name+"-"+e_name+"-MT=5"
                    xs_dict[long_name] = xs

            # get the normal reactions, found in mf=3
            for mt, rx in inc_f.reactions.items():
                if any([(mt in AMBIGUOUS_MT), (mt in FISSION_MTS), (301<=mt<=459)]):
                    continue # skip the cases of AMBIGUOUS_MT, fission mt, and heating information. They don't give us useful information about radionuclides produced.

                append_name_list, xs_list = extract_xs(inc_f.atomic_number, inc_f.mass_number, rx, tabulated=True)
                # add each product into the dictionary one by one.
                for name, xs in zip(append_name_list, xs_list):
                    xs_dict[nuc_sort_name + '-' + name] = xs

    # memory management
    print("Deleting decay_dict and endf_file_list since it will no longer be used in this script, in an attempt to reduce memory usage")
    del decay_dict; del endf_file_list; gc.collect()

    xs_dict = sort_and_trim_ordered_dict(xs_dict)

    print("\nCollapsing the cross-section to the group structure specified by 'gs.csv' and then saving it as 'response.csv' ...")
    sigma_df = collapse_xs(xs_dict, gs)
    del xs_dict; gc.collect()

    if not (SHOW_SEPARATE_MT_REACTION_RATES:=False):
        sigma_df = merge_identical_parent_products(sigma_df)

    if SORT_BY_REACTION_RATE:
        sigma_df = sigma_df.loc[ary(sigma_df.index)[np.argsort(sigma_df.values@apriori)[::-1]]]
    print("Saving the cross-sections to file as 'response.csv'...")
    sigma_df.to_csv(os.path.join(sys.argv[-1], 'response.csv'))
    # saves the number of radionuclide produced per (neutron cm^-2) of fluence flash-irradiated in that given bin.

    # save parameters at the end.
    parameters_dict = dict(HPGe_eff_file=HPGe_eff_file, gamma_E=gamma_E, FISSION_MTS=FISSION_MTS, AMBIGUOUS_MT=AMBIGUOUS_MT, SORT_BY_REACTION_RATE=SORT_BY_REACTION_RATE, SHOW_SEPARATE_MT_REACTION_RATES=SHOW_SEPARATE_MT_REACTION_RATES)
    parameters_dict.update({sys.argv[0]+" argv": sys.argv[1:]})
    save_parameters_as_json(sys.argv[-1], parameters_dict)
    print("SUCCESS!")