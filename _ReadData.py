# typical system/python stuff
import os, sys
import json
from collections import OrderedDict
from tqdm import tqdm
# typical python numerical stuff
from numpy import array as ary; import numpy as np
from numpy import log as ln
import pandas as pd
# openmc stuff
import openmc
from openmc.data import IncidentNeutron, Decay, Evaluation
from openmc.data.reaction import REACTION_NAME
# uncertainties
import uncertainties
from uncertainties.core import Variable
# custom modules
from flux_convert import Integrate
from misc_library import haskey, welcome_message, HPGe_efficiency_curve_generator, MT_to_nuc_num

FISSION_MTS = (18, 19, 20, 21, 22, 38)
AMBIGUOUS_MT = (1, 3, 5, 18, 27, 101, 201, 202, 203, 204, 205, 206, 207, 649)

photopeak_eff_curve = HPGe_efficiency_curve_generator('.')
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
    modes = []
    for mode in dec_file.modes:
        modes.append(dict(daughter=mode.daughter, branching_ratio=mode.branching_ratio))
    return dict(decay_constant=decay_constant, modes=modes, spectra=dec_file.spectra)

def condense_spectrum(dec_file):
    """
    We will explicitly ignore all continuous distributions because they do not show up as clean gamma lines.
    Note that the ENDF-B/decay/ directory stores a lot of spectra (even those with very clear lines) as continuous,
        so it may lead to the following method ignoring it.
        The workaround is just to use another library where the evaluators aren't so lazy to not use the continuous_flag=='both' :/
    """
    count = Variable(0.0,0.0)
    if haskey(dec_file['spectra'], 'gamma') and haskey(dec_file['spectra']['gamma'], 'discrete'):
        norm_factor = dec_file['spectra']['gamma']['discrete_normalization']
        for gamma_line in dec_file['spectra']['gamma']['discrete']:
            if np.clip(gamma_line['energy'].n, *gamma_E)==gamma_line['energy'].n:
                count += photopeak_eff_curve(gamma_line['energy']) * gamma_line['intensity']/100 * norm_factor
    if haskey(dec_file['spectra'], 'xray') and haskey(dec_file['spectra']['xray'], 'discrete'):
        norm_factor = dec_file['spectra']['xray']['discrete_normalization']
        for xray_line in dec_file['spectra']['xray']['discrete']:
            if np.clip(xray_line['energy'].n, *gamma_E)==xray_line['energy'].n:
                count += photopeak_eff_curve(xray_line['energy']) * xray_line['intensity']/100 * norm_factor
    del dec_file['spectra']
    dec_file['countable_photons'] = count #countable photons per parent
    return

def deduce_daughter_from_mt(parent_atomic_number, parent_atomic_mass, mt):
    """
    Given the atomic number and mass number, get the daughter in the format of 'Ag109'.
    """
    if 50<mt<=91:
        element_symbol = openmc.data.ATOMIC_SYMBOL[parent_atomic_number]
        product_mass = str(parent_atomic_mass)
        return element_symbol+product_mass+f'_m{mt-50}'
    element_symbol = openmc.data.ATOMIC_SYMBOL[parent_atomic_number + MT_to_nuc_num[mt][0]]
    product_mass = str(parent_atomic_mass + MT_to_nuc_num[mt][1])
    if len(MT_to_nuc_num[mt])>2 and MT_to_nuc_num[mt][2]>0: # if it indicates an excited state
        excited_state = '_m'+str(MT_to_nuc_num[mt][2])
        return element_symbol+product_mass+excited_state
    else:
        return element_symbol+product_mass

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
    if len(rx_file.products)==0:
        name = deduce_daughter_from_mt(parent_atomic_number, parent_atomic_mass, rx_file.mt)+'-MT='+str(rx_file.mt) # deduce_daughter_from_mt will return the ground state value
        appending_name_list.append(name)
        xs_list.append(detabulate(xs) if (not tabulated) else xs)
    else:
        for prod in rx_file.products:
            appending_name_list.append(prod.particle.replace('_e', '_m')+'-MT='+str(rx_file.mt))
            partial_xs = openmc.data.Tabulated1D(xs.x, prod.yield_(xs.x) * xs.y,
                                                breakpoints=xs.breakpoints, interpolation=xs.interpolation)
            xs_list.append(detabulate(partial_xs) if (not tabulated) else partial_xs)
    return appending_name_list, xs_list

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

def serialize_dict(mixed_object):
    if isinstance(mixed_object, dict):
        for key, val in mixed_object.items():
            mixed_object[key] = serialize_dict(val)
    elif isinstance(mixed_object, list):
        for ind, item in enumerate(mixed_object):
            mixed_object[ind] = serialize_dict(item)
    elif isinstance(mixed_object, uncertainties.core.AffineScalarFunc):
        mixed_object = str(mixed_object) # rewrite into str format
    else: # can't break it down, and probably is a scalar.
        pass
    return mixed_object

def clean_up_missing_records(xs_dict, decay_dict):
    """
    If there are entries in the xs_dict that gives a very unstable product (e.g. Ag_m22) as a primary product:
        so unstable such that there ISN'T a record of it in the decay_dict, then we can only assume that it will immediately decay to the ground state of that isotope as soon as it's produceds.
    """
    print("Cleaning up the fast-decaying isotopes by assuming that they decay directly to the ground state:")
    for parent_product_mt in tqdm(list(xs_dict.keys())):
        if parent_product_mt.split('-')[1] not in decay_dict.keys(): # product not recorded as an isotope in decay_dict
            parent, product, long_mt = parent_product_mt.split('-')
            new_product = product.split('_')[0]
            if new_product in decay_dict.keys():
                new_name = '-'.join([parent, new_product, long_mt])
                if new_name in xs_dict.keys():
                    # merge with existing xs record
                    xs2 = xs_dict[new_name] # expand out the xs2 in the next step. Assuming xs2 and xs would have the same group structure.
                    xs_dict[new_name] = openmc.data.Tabulated1D(xs2.x, xs2.y+xs_dict[parent_product_mt].y, breakpoints=xs2.breakpoints, interpolation=xs2.interpolation) # modify existing entry
                else:
                    xs_dict[new_name] = xs_dict[parent_product_mt] # create entirely new xs entry
                del xs_dict[parent_product_mt] # remove the old, unstable parent entry.

def collapse_xs(xs_dict, gs_ary):
    """
    Calculates the group-wise cross-section in the given group-structure
    by averaging the cross-section within each bin.
    """
    collapsed_sigma = dict()
    print("Collapsing the cross-sections to the correct group-structure...")
    for parent_product_mt, xs in tqdm(xs_dict.items()):
        I = Integrate(xs)
        collapsed_sigma[parent_product_mt] = ary([I(*gs_ary[i])/np.diff(gs_ary[i]) for i in range(len(gs_ary))]).flatten()
    return pd.DataFrame(collapsed_sigma).T

if __name__=='__main__':
    assert os.path.exists(os.path.join(sys.argv[-1], 'gs.csv')), "Output directory must already have gs.csv"
    gs = pd.read_csv(os.path.join(sys.argv[-1], 'gs.csv')).values
    if (SORT_BY_REACTION_RATE:=True):
        assert os.path.exists(os.path.join(sys.argv[-1], 'integrated_apriori.csv')), "Output directory must already have integrated_apriori.csv in order to sort the response.csv in descending order of expected-radionuclide-population later on."
        apriori = pd.read_csv(os.path.join(sys.argv[-1], 'integrated_apriori.csv'))['value'].values
    endf_file_list = welcome_message()
    print(f"Loaded {len(endf_file_list)} different mateiral files,\n")

    # First compile the decay records
    print("Compiling the decay_dict...")
    decay_dict = OrderedDict()
    for file in tqdm(endf_file_list):
        if repr(file).strip('<>').startswith('Radio'): # applicable to materials with (mf, mt) = (8, 457) file section
            dec_f = Decay.from_endf(file)
            decay_dict[str(dec_f.nuclide['atomic_number']).zfill(3)+dec_f.nuclide['name']] = extract_decay(dec_f)
    decay_dict = sort_and_trim_ordered_dict(decay_dict) # reorder it so that it conforms to the 
    
    print("Condensing each decay spectrum...")
    for name, dec_file in tqdm(decay_dict.items()):
        condense_spectrum(dec_file)

    # Then compile the Incident-neutron records
    print("Compiling the raw cross-section dictionary ...")
    xs_dict = OrderedDict()
    for file in tqdm(endf_file_list):
        if repr(file).strip('<>').startswith('Incident-neutron'):
            inc_f = IncidentNeutron.from_endf(file)
            nuc_sort_name = str(inc_f.atomic_number).zfill(3)+inc_f.name

            for mt, rx in inc_f.reactions.items():
                xs_raw = rx.xs['0K']
                
                # ignore the entries that isn't telling us the cross-section (which are usually mf=2 files which tells us about resonances, which I don't care)
                if not hasattr(xs_raw, 'x'):
                    continue # skip the mt's that doesn't actually give us the cross-sections
                if any([(mt in AMBIGUOUS_MT), (mt in FISSION_MTS), (301<=mt<=459)]):
                    continue # skip the cases of AMBIGUOUS_MT, fission mt, and heating information. They don't give us useful information about radionuclides produced.

                append_name_list, xs_list = extract_xs(inc_f.atomic_number, inc_f.mass_number, rx, tabulated=True)
                for name, xs in zip(append_name_list, xs_list):
                    xs_dict[nuc_sort_name+'-'+name] = xs
    clean_up_missing_records(xs_dict, decay_dict)
    xs_dict = sort_and_trim_ordered_dict(xs_dict)


    sigma_df = collapse_xs(xs_dict, gs)
    if SORT_BY_REACTION_RATE:
      sigma_df = sigma_df.loc[ary(sigma_df.index)[np.argsort(sigma_df.values@apriori)[::-1]]]
    sigma_df.to_csv(os.path.join(sys.argv[-1], 'response.csv'))
    # saves the number of radionuclide produced
    #   per (neutron cm^-2) of fluence flash-irradiated in that given bin.

    with open(os.path.join(sys.argv[-1], 'decay_records.json'), 'w') as f:
        decay_dict = serialize_dict(decay_dict)
        json.dump(decay_dict, f)

    """
    See _check_stuff.py for the to-do list. 2020-12-06 15:50:18
    """