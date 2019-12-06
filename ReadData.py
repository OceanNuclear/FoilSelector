import openmc
from numpy import array as ary
MT = { 'inc':3, 'decay':8, 'inc_covar':33, 'decay_covar':40 , 'general_info':1, 'resonance_params':2, 'decay_multiplicities':9, 'radionuclide_production_cross_section':10}
try:
    import os, sys
    folder_list = sys.argv[1:]
except IndexError:
    print("usage:")
    print("'python "+sys.argv[0]+" folders/ containing/ endf/ files/ in/ descending/ order/ of/ priority/'")
    print("Use wildcard (*) if necessary")
    print("The endf files in question They can be downloaded by running the make file.")
    # from https://www.oecd-nea.org/dbforms/data/eva/evatapes/eaf_2010/ and https://www-nds.iaea.org/IRDFF/
    # Currently does not support reading h5 files yet, because openmc does not support reading ace/converting endf into hdf5/reading endf using enjoy yet
    exit()
print(f"Reading from {len(folder_list)} folders,")
file_list = []
for folder in folder_list:
    file_list += [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.endf')]
print(f"Found {len(file_list)} regular files ending in '.endf' ...")
#read in each file:
endf_data = []
for path in file_list:
    endf_data += openmc.data.get_evaluations(path)
'''
Must use openmc.data.get_evaluations in this manner( using data.endf.Evaulation('eaf/*.endf') ), because:
1. openmc.data.get_evaluations can't read the covariance data left in the ```cat eaf/*.endf > all_eaf``` file
    (EAF data is similar enough to endf file that it can be read like a endf file on its own; but once EAF files are concatenated together, you can't read it like an ENDF file anymore.)
2. openmc.data.IncidentNeutron will lead to 743 lines being ignored in the IRDFF file:
    (MT=[3,23,11,4,3,11,9,3,21,9,3,31,11,3,17,9,3,17,9,3,25,11,3,25,11,3,9,9,3,16,9,3,13,9,3,11,9,3,13,9,3,15,9,316,])

'''
print(f"Loaded {len(endf_data)} different material files,") #should be a list
print()

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

# Print preliminary analysis to see how much of that data is unique.
print("################Preliminary data analysis################")
# get all the openmc.data.Evaluation objects' names,
'''
all_MF={iso:set() for iso in gnd_name_list}
all_MT={iso:set() for iso in gnd_name_list}
for i in range(len(endf_data)):
    dat = endf_data[i]
    rl = ary(dat.reaction_list)
    for row in rl:
        all_MF[ dat.gnd_name ].add(row[0])
        all_MT[ dat.gnd_name ].add(row[1])
'''

incident_neutron = [iso.gnd_name for iso in endf_data if (MT['inc']         in ary(iso.reaction_list)[:,0])] # Get the list of files containing Incident-neutron data
neutron_covar    = [iso.gnd_name for iso in endf_data if (MT['inc_covar']   in ary(iso.reaction_list)[:,0])] # 
decay            = [iso.gnd_name for iso in endf_data if (MT['decay']       in ary(iso.reaction_list)[:,0])] # 
decay_covar      = [iso.gnd_name for iso in endf_data if (MT['decay_covar'] in ary(iso.reaction_list)[:,0])] # 
prod_cross_sec   = [iso.gnd_name for iso in endf_data if (MT['radionuclide_production_cross_section'] in ary(iso.reaction_list)[:,0])] # 
overlapping_isomer_set = list(set([ iso for iso in incident_neutron if (iso in decay) ]))
fully_equipped_isomer_set = list(set([ iso for iso in incident_neutron if (iso in neutron_covar) and (iso in decay) and (iso in decay_covar) and (iso in prod_cross_sec) ]))
zero_mass_isomer = [ _sort_name for _sort_name in repr_name_list if _sort_name.split("-")[-1]=='0' ]
print(f"{len(incident_neutron)} files containing incident neutron data, spread over {len(set(incident_neutron))} isomers were found,")
print(f"{len(neutron_covar)} files containing incident neutron data's covariances, spread over {len(set(neutron_covar))} isomers were found,")
print(f"{len(prod_cross_sec)} files containing radionuclide production cross sections, spread over {len(set(prod_cross_sec))} isomers were found,")
print()
print(f"{len(decay)} files containing decay data, spread over {len(set(decay))} isomers were found,")
print(f"{len(decay_covar)} files containing decay data's covariances, spread over {len(set(decay_covar))} isomers were found,")
print()
print(f"{len(overlapping_isomer_set)} isomers were found to have data of both incident neutrons and decay data; while")
print(f"{len(fully_equipped_isomer_set)} isomers were found to have data of all five data types (mf numbers) listed above.")
print(f"{(n:=len(set(zero_mass_isomer)))} natural composition elements (i.e. isomers with mass number=0) were found,")
print(" "*(len(str(n))+1)+"and will be used with the highest priority when the user does not specify the isotopic composition.")

print(f"Otherwise, the priority follows the order that the files were read in.")
# This allows sorting by element, and then by mass number.
for i in range(len(endf_data)):
    endf_data[i]._sort_name = sorting_names[i]
    # endf_data[n].info['identifier'] gives the database ('EAF' vs not 'EAF'('IRDFF' or 'ENDF-V/B')) and MAT number.
#Turn it into a dictionary using dict comprehension:
iso_dict = { gnd_name:[ _eval_obj for _eval_obj in endf_data if _eval_obj.gnd_name==gnd_name ] for gnd_name in gnd_name_list }
# may lead to repeated data entry, but that's okay, because we're using a dictionary.
# Check how to get the microscopic cross-section
# load in the table the table of where is the natural mixture
# since ENDF contain no data about the abundances.