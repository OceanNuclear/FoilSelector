import openmc
import time 
from numpy import array as ary; import numpy as np
import json, pickle
import os, sys
# import logging
# logHanlder = logging.Handler()
# logging.addHandler(logHanlder)
'''
Purpose:
    Read in all nuclides from their endf files into a format that we care about (a Reaction instance),
        from which we can quickly extract information about.
    The reaction's sigma WON'T be stored if we don't know ALL of its product's decay information.
    
    We can only assume that no alpha/neutrons will be emitted by that nuclide after neutron irradiation
        by its un-documented decay products by the time it gets into the HPGe,
        Since these decay products are likely fast-decaying.
Exclusions:
    Try to exclude atoms that can fission.
'''
EXCLUDE_FISSION = True
MF = { 'inc':3, 'decay':8, 'inc_covar':33, 'decay_covar':40 , 'general_info':1, 'resonance_params':2, 'decay_multiplicities':9, 'radionuclide_production_cross_section':10}
def welcome_message():
    try:
        folder_list = sys.argv[1:]
        assert len(folder_list)>0
        #Please update your python to 3.6 or later.
        print(f"Reading from {len(folder_list)} folders,")
        file_list = []
        for folder in folder_list:
            file_list += [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.endf')]
    except (IndexError, AssertionError):
        print("usage:")
        print("'python "+sys.argv[0]+" folders/ containing/ endf/ files/ in/ descending/ order/ of/ priority/ [output/]'")
        print("where the outputs-saving directory 'output/' is only requried when read_apriori_and_gs_df is used.")
        print("Use wildcard (*) if necessary")
        print("The endf files in question can be downloaded")
        # by running the make file.")
        # from https://www.oecd-nea.org/dbforms/data/eva/evatapes/eaf_2010/ and https://www-nds.iaea.org/IRDFF/
        # Currently does not support reading h5 files yet, because openmc does not support reading ace/converting endf into hdf5/reading endf using enjoy yet
        sys.exit()

    print(f"Found {len(file_list)} regular files ending in '.endf' ...")
    #read in each file:
    endf_data = []
    for path in file_list:
        endf_data += openmc.data.get_evaluations(path)
    return endf_data
    
'''
Must use openmc.data.get_evaluations in this manner( using data.endf.Evaulation('eaf/*.endf') ), because:
1. openmc.data.get_evaluations can't read the covariance data left in the ```cat eaf/*.endf > all_eaf``` file
    (EAF data is similar enough to endf file that it can be read like a endf file on its own; but once EAF files are concatenated together, you can't read it like an ENDF file anymore.)
2. openmc.data.IncidentNeutron will lead to 743 lines being ignored in the IRDFF file:
    (MT=[3,23,11,4,3,11,9,3,21,9,3,31,11,3,17,9,3,17,9,3,25,11,3,25,11,3,9,9,3,16,9,3,13,9,3,11,9,3,13,9,3,15,9,316,])

'''

def read_reformatted(working_dir):
    with open(os.path.join(working_dir, 'reactions.pkl'), 'rb') as f:
        rdict = pickle.load(f)
    with open (os.path.join(working_dir, 'decay_radiation.pkl'), 'rb') as f:
        dec_r = pickle.load(f)
    with open (os.path.join(working_dir, 'all_mts.json'), 'r') as f:
        all_mts = json.load(f)
    return rdict, dec_r, all_mts

def save_reformatted(rdict, dec_r, all_mts, working_dir):
    with open(os.path.join(working_dir, 'reactions.pkl'), 'wb') as f:
        pickle.dump(rdict, f)
    with open (os.path.join(working_dir, 'decay_radiation.pkl'), 'wb') as f:
        pickle.dump(dec_r, f)
    with open (os.path.join(working_dir, 'all_mts.json'), 'w') as f:
        json.dump(all_mts, f)
    return

class Reaction:
    def __init__(self, openmcRlist, openmc_eval_list, rcomplist, resonances_list, kTs_list):
        #did not load resonance covariances because it likes to misbehave
        assert len(openmcRlist)>0
        mt = openmcRlist[0].mt
        #Things that I'd only take the first value for as a representative value.
        # for v in openmcRlist[0].values():
        #     self.sigma = v #choose the last cross-section, i.e. the one at the highest temperature.
        self.sigma = openmcRlist[0].xs['0K']
        self.kTs = kTs_list[0]
        self.q_value = openmcRlist[0].q_value

        #Things that I'd save the complete list for
        self.alternatives = openmcRlist #still save the whole list
        
        #Things taht I'd sum over the entire list 
        self.redundant = any([r.redundant for r in openmcRlist])
        self.products = [ r.products for r in openmcRlist ]
        self.products += [ r.derived_products for r in openmcRlist ]
        self.reaction_component_lists = rcomplist
        self.resonances_list = resonances_list        

        #Read the openmc_eval_list's relevant sections
        self.products_name = []
        self.parent = openmc_eval_list[0].target
        matching_section = (8, mt)
        for eval_file in openmc_eval_list: #TERRIBLE bodge to read backwards to find out the relevant MF=8 file from the Evaluation object's section attribute.
            if matching_section in eval_file.section.keys():
                second_line_onwards = eval_file.section[matching_section].split("\n ")[1:]
                for line in second_line_onwards:
                    ZA = int(float(line.split()[0].replace("+", "e")))
                    gnd_name = openmc.data.gnd_name( ZA//1000, ZA%1000, int(line.split()[3]) )
                    self.products_name.append(gnd_name)

def get_names(endf_data):
    # Sorting the gnd_names into order:
    gnd_name_list = [iso.gnd_name for iso in endf_data] # get all the isotope names
    eval_names = [ repr(iso) for iso in endf_data ]
    repr_name_list = [ iso.replace("Incident-neutron","Incident neutron").split()[4] for iso in eval_names]# and trim them.
    sorting_names = [ "-".join([ iso.split("-")[i].zfill(3-2*i) for i in range(3)]) for iso in repr_name_list]
    sorting_names, gnd_name_list = [ list(t) for t in zip(*sorted(zip(sorting_names, gnd_name_list))) ]
    #print information about usable number of isomers
    print("There are", len(endf_data)-len(set(eval_names)),"repeated entries.")
    print(f"{len(set(gnd_name_list))} unique isomers were found.")
    print()
    return gnd_name_list, repr_name_list, sorting_names

def analyse_numbers_and_overlaps(endf_data, repr_name_list):
    # Print preliminary analysis to see how much of that data is unique.
    print("################Preliminary data analysis################")
    incident_neutron = [iso.gnd_name for iso in endf_data if (MF['inc']         in ary(iso.reaction_list)[:,0])] # Get the list of files containing Incident-neutron data
    neutron_covar    = [iso.gnd_name for iso in endf_data if (MF['inc_covar']   in ary(iso.reaction_list)[:,0])] # 
    decay            = [iso.gnd_name for iso in endf_data if (MF['decay']       in ary(iso.reaction_list)[:,0])] # 
    decay_covar      = [iso.gnd_name for iso in endf_data if (MF['decay_covar'] in ary(iso.reaction_list)[:,0])] # 
    prod_cross_sec   = [iso.gnd_name for iso in endf_data if (MF['radionuclide_production_cross_section'] in ary(iso.reaction_list)[:,0])] # 
    overlapping_isomer_set = list(set([ iso for iso in incident_neutron if (iso in decay) ]))
    fully_equipped_isomer_set = list(set([ iso for iso in incident_neutron if (iso in neutron_covar) and (iso in decay) and (iso in decay_covar) and (iso in prod_cross_sec) ]))
    zero_mass_isomer = [ _sort_name for _sort_name in repr_name_list if _sort_name.split("-")[-1]=='0' ]
    print(f"{len(incident_neutron)} files containing incident neutron data, spread over {len(set(incident_neutron))} isomers were found,")
    print(f"{len(neutron_covar)} files containing incident neutron data's covariances, spread over {len(set(neutron_covar))} isomers were found,")
    print(f"{len(prod_cross_sec)} files containing radionuclide production cross sections, spread over {len(set(prod_cross_sec))} isomers were found,")
    print()
    print(f"{len(decay)} files containing decay data (either from incident neutrons, or directly), spread over {len(set(decay))} isomers were found,")
    print(f"{len(decay_covar)} files containing decay data's covariances, spread over {len(set(decay_covar))} isomers were found,")
    print()
    print(f"{len(overlapping_isomer_set)} isomers were found to have data of both incident neutrons and decay data; while")
    print(f"{len(fully_equipped_isomer_set)} isomers were found to have data of all five data types (mf numbers) listed above.")
    print(f"{(n:=len(set(zero_mass_isomer)))} natural composition elements (i.e. isomers with mass number=0) were found,")
    #Please update your python to 3.8 or later.
    print(" "*(len(str(n))+1)+"and will be used with the highest priority when the user does not specify the isotopic composition.")

    print(f"Otherwise, the priority follows the order that the files were read in.")
    return incident_neutron, neutron_covar, decay, decay_covar, prod_cross_sec, overlapping_isomer_set, fully_equipped_isomer_set

def get_all_mt(iso_dict):
    starttime = time.time()
    # local dictionary, i.e. there are as many of these are there are isotopes.
    inc_r, dec_r = {}, {} #stores the entire reaction object
    inc_mt, rcomp, kTs, resonances = {}, {}, {}, {} # stores ony the mt number, the reaction_components, 
    # returns these information when given the gnd_name (and in other cases mt numebr).
    
    #global dictionary
    all_mts = {}
    rcomplist = {}
    for gnd_name, data_list in iso_dict.items():
        inc_r[gnd_name] = {} #create empty dict ready to receive the openmc.data.IncidentNeutron objects
        inc_mt[gnd_name] = set() #create emtpy set ready to receive the mt numbers
        rcomp[gnd_name] = {}
        resonances[gnd_name], kTs[gnd_name] = [], []

        for data in data_list:
            if "Incident" in repr(data):
                inc = openmc.data.IncidentNeutron.from_endf(data)
                for r in inc.reactions.values():
                    inc_mt[gnd_name].add(r.mt)
                    all_mts[int(r.mt)] = repr(r)[11:-1].split(" ")[-1]
                    if not ( r.mt in inc_r[gnd_name].keys() ): inc_r[gnd_name][r.mt] = [] #if no corresponding list, then start one.
                    inc_r[gnd_name][r.mt].append(r)
                    if not ( r.mt in rcomp[gnd_name].keys() ): rcomp[gnd_name][r.mt] = [] #if no corresponding list, then start one.
                    rcomp[gnd_name][r.mt] += inc.get_reaction_components(r.mt)
                kTs[gnd_name].append(inc.kTs)
                resonances[gnd_name].append(inc.resonances)
            else:
                dec = openmc.data.Decay.from_endf(data)
                if not gnd_name in dec_r.keys():
                    dec_r[gnd_name] = []
                dec_r[gnd_name].append(dec)
    print(f"taken {time.time()-starttime} s to convert the data from Evaulation() objects into IncidentNeutron()/ Decay() objects")
    return inc_r, inc_mt, all_mts, rcomp, resonances, kTs, dec_r
translation = {
    2:(0, 0),
    4:(0, 0),
    11:(-1,-3),
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

def main_read():
    endf_data = welcome_message()
    print(f"Loaded {len(endf_data)} different mateiral files,\n")
    # get all the openmc.data.Evaluation objects' names,
    gnd_name_list, repr_name_list, sorting_names = get_names(endf_data)
    
    analyse_numbers_and_overlaps(endf_data, repr_name_list)

    # This allows sorting by element, and then by mass number.
    # #Give it a name for sorting purpose.
    # for i in range(len(endf_data)):
    #     endf_data[i]._sort_name = sorting_names[i]
        # endf_data[n].info['identifier'] gives the database ('EAF' vs not 'EAF'('IRDFF' or 'ENDF-V/B')) and MAT number.
    #Turn it into a dictionary using dict comprehension:
    iso_dict = { gnd_name:[ eval_obj for eval_obj in endf_data if eval_obj.gnd_name==gnd_name ] for gnd_name in gnd_name_list } # may lead to repeated data entry, but that's okay, because we're using a dictionary.
    
    inc_r, inc_mt, all_mts, rcomp, resonances, kTs, dec_r = get_all_mt(iso_dict) # These information is necessary for initializing the Reaction objects in the rdict.
    '''
    #inc_r[gnd_name][mt] = openmc.data.IncidentNeutron.from_endf(Evaulation_file_obj).reaction[mt]
    #inc_mt[gnd_name] = set(all_reactions_mts_for_that_isomer)
    #all_mts[mt]= set(all_reaction_mts)
    '''
    rdict = {}
    starttime = time.time()
    for gnd_name, data_collection in iso_dict.items():
        rdict[gnd_name]={ mt:Reaction(inc_r[gnd_name][mt], data_collection, rcomp[gnd_name][mt], resonances[gnd_name], kTs[gnd_name]) for mt in inc_mt[gnd_name] }
        if EXCLUDE_FISSION:
            if 18 in rdict[gnd_name].keys():
                rdict[gnd_name]={}
                print("Excluded", gnd_name, "as it is fissionable/fissile")
    print(f"taken {time.time()-starttime} s to convert the evaluation objects into Reaction() objects")

    return rdict, dec_r, all_mts

if __name__=='__main__':
    rdict, dec_r, all_mts = main_read()
    save_reformatted(rdict, dec_r, all_mts, sys.argv[-1])

# Next thing to do: deal with the cases where product_num isn't present/is wrong.
# Turns out all IncidentNeutron objects cannot have reaction 457 stored inside them.

'''
objective (kept here for future development use; currently this can be ignored.)
1. Given the database, get all isomers, then sorted by , aand then by a specific order/ priority. 
1.1 Check if the covariance line has anything, if so, grab that raw file section too.
1.2 convert to IncidentNeutron data and obtain the cross-section from .xs['0K'].x and .xs['0K'].y
2. Get reaction products from the MF=8 file for each reaction; if not, deduce it. (It seems like I never have to deduce any.)
3. Search for the relevant openmc.data.Decay spectrum
4. Return the interested isomer's all relevant xs.
'''