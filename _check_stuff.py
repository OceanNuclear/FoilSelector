import openmc.data
from misc_library import load_endf_directories
import sys
from matplotlib import pyplot as plt
import numpy as np
import json
sdir = lambda x: [i for i in dir(x) if '__' not in i]
plot_tab = lambda tab, **kwargs: plt.plot(*[getattr(tab, ax) for ax in 'xy'], **kwargs)
plot_delta = lambda E, height, dE, **kwargs: plt.plot([E-dE/2, E, E+dE/2], [0, height, 0], **kwargs)
haskey = lambda dict_instance, key: key in dict_instance.keys()
from collections import defaultdict
from misc_library import openmc_variable
vars().update(openmc_variable)
"""
This is a data exploratory file.

OBJECTIVES:
12. Compare with experimental results: check dimensionality etc.
2. Check to make sure that every library uses products[*].yield_.interpolation==2 (CHECK_YIELD)
16. Write Adder and Multiplier for Tabulated1D (to accurately obtain the "partial")
-1. Plot the gamma peaks and find overlaps
13. Fix scheme 5 (test fails frequently, need to print the errors in more detail to figure out where they're coming from.)
17. Rewrite Foil and FoilSet with the SOLID principle (inheritence: CLOPEN)
18. Clean up flux_convert.py?

NOTE:
    # note that sometimes yield > unity. And I don't know why. I'm just living with it like this for the moment.
    # It is apparent that the (MF=10, MT!=5) files stores information that can also be extracted by multiplying the yield_ onto the MF=3 file's cross-section.
        So there's no need to read all of the (MF=10, MT!=5) files.

DONE in the past 3 days (2021-01-18 13:28:19):
# What am I going to do with the other MF=10 files?
# why doesn't clean_up_missing_records throw an error when adding tab1's of dfferent shapes?
# 7.5 OR rewrite openmc to make it read higher than 30 MeV.
# Big problem: The Au197m file gives the exact same info as the Au197 file!! Same reactions, same cross-sections, but openmc doesn't alert us of that!
# Write bash script/Python script combination that uses extract_xs_endf
#     Create the following files:
#     aafiles points to aafluxes
#     aafluxes is a dummy file with n-1 zeros, 1 1, 1 wall-load, 1 comment line.
#     bash file to call python file to get the entire list of reactions available in TENDL (turn _m1 into m, _m2 into n, ignore everyone else?) and put it into a file.
#     'extract_xs_endf Sb105-Sn105-MT=103 n 1102 Sb105 103 Sn105 aafiles'
#     bash file takes the 5th column of .out and then append it into the 'fispact-response.csv' horizontally; then delete the .out and .log file.
#     Convert flux into the relevant group structure and then save it as 'rebinned_flux.csv'.
# Idea ignored since FISPACT has now been proven as useless
DONE: 
# 7. use the extract_xs_endf executable from FISPACT
#     https://fispact.ukaea.uk/wiki/Reaction_extract
#     Find out if I'm misusing extract_xs_endf
#         ( record clearly exists (grep '10  5 3107' /home/ocean/Documents/PhD/FoilSelector/TENDL/gxs-1102/Au197g.asc) )
# -2. Caclulate rname| volume| thicknesses
# 3. Figure out how to treat continuous_flag=='both'/'continuous' distribution of gamma. (CHECK_DECAY_CONT)
# 5. Change the decay_dict[...]['mode'] from [{"daughter":"Al27", "branching_ratio":1.0+/-0.0}] to {"Al27":1.0+/-0.0} and make the rest of the handling pretty.
# 4. Speed up the integration by taking advantage of numpy
# 9. Rewrite flux_convert to use detabulate instead of pkl
# 8. Incorperate the Integrate class into flux_convert.py and _ReadData.py
# 11. Create a preprocessing_data/ directory for universally used data (efficiency, material densities, etc.) to sit in.
# 2.5 Check products[*].yield_.y<=1.0 (CHECK_YIELD)
# 1. Figure out how to treat the unmatched E_max and E_min energy cases. (CHECK_LIMITS) # Just let them be.
# 8.5 redo with faster and larger library.
# 10. Write unittest for Integrate

Deprecated jobs:
# 6. change the count rates 0.0+/-nan to 0.0+/-0.0
# 16. To make it usable across many libraries, add a check to make sure the MF10MT5 reactions CORRESPONDS to and compliments at least one chopped off reaction?
    Not needed because we can just expect the data to be stored in the correct way.
# 15. If MF10, MT!=5 file is present, choose that over the "multiplying by the yield_" method for extracting the partial cross-section?
    Nah, easier to just deal with just the yield without the branching if conditions.
"""

CHECK_MAX_E = False
CHECK_YIELD = True
CHECK_LIMITS = True
CHECK_DECAY_CONT = False
if __name__=='__main__':
    endf_file_list = load_endf_directories(sys.argv[1:])
    if any([CHECK_MAX_E, CHECK_YIELD, CHECK_LIMITS]) :
        reactions = {f.gnd_name : openmc.data.IncidentNeutron.from_endf(f).reactions for f in endf_file_list if 'Incident' in repr(f)}
    if CHECK_DECAY_CONT:
        decay_dict = {f.gnd_name: openmc.data.Decay.from_endf(f) for f in endf_file_list if 'radio' in repr(f).lower()}

    if CHECK_MAX_E:
        for r_dict in reactions.values():
            for mt in r_dict.keys():
                desired_xs = r_dict[mt].xs['0K']
                r_dict[mt] = desired_xs.x.max() if hasattr(desired_xs, 'x') else np.nan

        LOWER_LIMIT_ON_MAX = 1E9
        for nuclide, r_dict in reactions.items():
            for mt in r_dict.keys():
                if r_dict[mt]<LOWER_LIMIT_ON_MAX:
                    print(f'{nuclide} has maximum E up record for {mt = } up to = {r_dict[mt]} eV only!' )

    if CHECK_YIELD:
        for gnd_name, r_dict in reactions.items():
            for mt in r_dict.keys():
                if mt in FISSION_MTS: # exclude fission
                    continue
                elif hasattr(r_dict[mt].xs['0K'], 'x'):
                    xs_x, xs_y = [ getattr(r_dict[mt].xs['0K'], ax) for ax in 'xy']
                    xs_interp = r_dict[mt].xs['0K'].interpolation
                    # if not np.logical_or(np.equal(xs_interp, 2), np.equal(xs_interp, 1)).all():
                    #     print(f"interpolation scheme is neither histogramic nor linear for {gnd_name} mt={mt}")
                else:
                    continue
                for ind, prod in enumerate(r_dict[mt].products):
                    assert hasattr(prod, 'yield_'), f"This product {prod} doesn't have a yield!"
                    interpolation = prod.yield_.interpolation
                    if not np.equal(interpolation, 2).all():
                        print(f"{gnd_name} {mt=} {prod.particle} interpolation scheme = {set(interpolation)} != (2) at ")
                    if (prod.yield_.y>1.0).any():
                        print(f"{gnd_name} {mt=} {prod.particle} has maximum yield={prod.yield_.y.max()}>1!")
                    # if prod.emission_mode != "prompt":
                    #     print(gnd_name, f"MT={mt}", "has emission mode = ", prod.emission_mode)

    if CHECK_LIMITS:
        for gnd_name, r_dict in reactions.items():
            for mt in r_dict.keys():
                if mt in FISSION_MTS:
                    continue
                if len(r_dict[mt].products)==0:
                    continue
                elif hasattr(r_dict[mt].xs['0K'], 'x'):
                    xs = r_dict[mt].xs['0K']
                else:
                    continue

                for func in ['min', 'max']:
                    extrema = {"total": getattr(xs.x, func)()}
                    for prod in r_dict[mt].products:
                        extrema[prod.particle] = getattr(prod.yield_.x, func)()
                    if not np.isclose(list(extrema.values())[0], list(extrema.values())[1:]).all():
                        print(gnd_name, f"{mt=}", func, extrema)
                    # if not np.isclose(yield_x.min(), xs_x.min()):
                    #     print(gnd_name, f"MT={mt} has mismatched {prod} energy range wrt. its xs,", yield_x.min(), xs_x.min())
                    # if not np.isclose(yield_x.max(), xs_x.max()):
                    #     print(gnd_name, f"MT={mt} has mismatched {prod} energy range wrt. its xs,", yield_x.max(), xs_x.max())

    if CHECK_DECAY_CONT:
        # continuous: all are 'gamma' radiation, and are induced by 'beta-' or 'sf'. They all are normalized (to a pretty good degree) probability distributions.
        
        all_continuous = defaultdict(dict)
        for parent, dec_file in decay_dict.items():
            if haskey(dec_file.spectra, 'xray') and dec_file.spectra['xray']['continuous_flag']!='discrete':
                all_continuous[parent]['xray'] = dec_file.spectra['xray']['continuous']
            if haskey(dec_file.spectra, 'gamma') and dec_file.spectra['gamma']['continuous_flag']!='discrete':
                all_continuous[parent]['gamma']= dec_file.spectra['gamma']['continuous']

        for name, cont_file in all_continuous.items():
            plt.title(name)
            for key in cont_file.keys():
                rel_prob = cont_file[key]['probability']
                abs_prob = openmc.data.Tabulated1D(rel_prob.x, rel_prob.y*decay_dict[name].spectra[key]['continuous_normalization'].n, rel_prob.breakpoints, rel_prob.interpolation)
                plot_tab(abs_prob, label=key+' '+str(cont_file[key]['type']))
            for rad_name, rad_file in decay_dict[name].spectra.items():
                if rad_name in ['xray', 'gamma'] and haskey(rad_file, 'discrete'):
                    for line in rad_file['discrete']:
                        plot_delta((line['energy']*rad_file['discrete_normalization']).n, line['intensity'].n, np.clip((line['energy']*rad_file['discrete_normalization'].n).s,1, 1000)) # centroid, height, width
            plt.legend()
            plt.yscale('log')
            plt.show()
        """ #checked and swa that they are all normalized to 1.
        # for key, val in continuous_only.items():
        for key, val in mixed_continuous.items():
            min_E, max_E = [getattr(val['probability'].x, minmax)() for minmax in ['min', 'max']]
            np.isclose(1, Integrate()(min_E, max_E))
        """