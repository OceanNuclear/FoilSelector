import openmc
from numpy import array as ary
from numpy import log as ln
from numpy import exp, sqrt
import numpy as np
if (DEV_MODE:=False):
    from matplotlib import pyplot as plt
import uncertainties
import json

haskey = lambda dict_instance, key: key in dict_instance.keys()

def unserialize_dict(dict_with_uncertainties):
    """
    Turn the string representation of the uncertainties back into uncertainties.core.Variable 's.
    """
    for key, val in dict_with_uncertainties.items():
        if isinstance(val, dict):
            unserialize_dict(val)
        elif isinstance(val, str):
            if '+/-' in val:
                if ')' in val:
                    multiplier = float('1'+val.split(')')[1])
                    val_stripped = val.split(')')[0][1:]
                else:
                    multiplier = 1.0
                    val_stripped = val
                dict_with_uncertainties[key] = uncertainties.core.Variable(*[float(i)*multiplier for i in val_stripped.split('+/-')])
        else:
            pass

def build_decay_chain(decay_parent, decay_dict, decay_constant_threshold=1E-23):
    """
    decay_parent is the potentially unstable nuclide.
    decay_parent
    """
    if not haskey(decay_dict, decay_parent):
        return_dict = {'name':decay_parent, 'decay_constant':uncertainties.core.Variable(0.0,0.0), 'countable_photons':Variable(0.0,0.0)}
        return 
    elif decay_dict[decay_parent]['decay_constant']<=decay_constant_threshold:
        parent = decay_dict[decay_parent]
        return_dict = {'name':decay_parent, 'decay_constant':parent['decay_constant'], 'countable_photons':parent['countable_photons']}
        return
    else:
        parent = decay_dict[decay_parent]
        return_dict = {'name':decay_parent, 'decay_constant':parent['decay_constant'], 'countable_photons':parent['countable_photons']}
        return_dict['modes'] = [{'daughter':build_decay_chain(mode['daughter'], decay_dict), 'branching_ratio':mode['branching_ratio']} for mode in parent['modes']]
        return return_dict

if __name__=='__main__':
    with open(os.path.join(sys.argv[-1], 'decay_records.json')) as f:
        decay_dict = json.load(f)
        unserialize_dict(decay_dict)
    sigma_df = pd.read_csv(os.path.join(sys.argv[-1], 'response.csv'), index_col=[0])
    for parent_product_mt in sigma_df.index:
        product = parent_product_mt.split('-')[1]
        product