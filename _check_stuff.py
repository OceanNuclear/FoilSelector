import openmc.data
from ReadData import welcome_message
import sys
from matplotlib import pyplot as plt
import numpy as np
import json
with open('ChipIR/all_mts.json') as f:
    all_mts = json.load(f)
sdir = lambda x: [i for i in dir(x) if '__' not in i]
plot_tab = lambda tab, **kwargs: plt.plot(*[getattr(tab, ax) for ax in 'xy'], **kwargs)
plot_delta = lambda E, height, dE, **kwargs: plt.plot([E-dE/2, E, E+dE/2], [0, height, 0], **kwargs)
haskey = lambda dict_instance, key: key in dict_instance.keys()
from collections import defaultdict
FISSION_MTS = (18, 19, 20, 21, 22, 38)
"""
Following the workflow:
8.5 redo with faster and larger library.
-1. Plot the gamma peaks and find overlaps
Exploratory file, with the current objectives of:
7. use the extract_xs executable from FISPACT
1. Figure out how to treat the unmatched E_max and E_min energy cases. (CHECK_LIMITS)
2. Check to make sure that every library uses products[*].yield_.interpolation==2 (CHECK_YIELD)
10. Write unittest for Integrate2 (which, BTW, should be renamed.)
12. Compare with experimental results

DONE: 
# -2. Caclulate rname| volume| thicknesses
# 3. Figure out how to treat continuous_flag=='both'/'continuous' distribution of gamma. (CHECK_DECAY_CONT)
# 5. Change the decay_dict[...]['mode'] from [{"daughter":"Al27", "branching_ratio":1.0+/-0.0}] to {"Al27":1.0+/-0.0} and make the rest of the handling pretty.
# 4. Speed up the integration by taking advantage of numpy
# 9. Rewrite flux_convert to use detabulate instead of pkl
# 8. Incorperate the Integrate class into flux_convert.py and _ReadData.py
# 11. Create a preprocessing_data/ directory for universally used data (efficiency, material densities, etc.) to sit in.

Deprecated jobs:
# 6. change the count rates 0.0+/-nan to 0.0+/-0.0
"""

CHECK_MAX_E = False
CHECK_YIELD = False
CHECK_LIMITS = True
CHECK_DECAY_CONT = False
if __name__=='__main__':
    endf_file_list = welcome_message()
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
                        print("interpolation scheme != 2 at ", f"{gnd_name} [MT={mt}]")
                    # if '_' in prod.particle :
                    #     print(gnd_name, mt, prod.particle)
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
                    extrema = ([getattr(tabulated, func)() for tabulated in [xs.x, *[prod.yield_.x for prod in r_dict[mt].products]]])
                    if not np.isclose(extrema[0], extrema[1:]).all():
                        print(gnd_name, mt, extrema)
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