from numpy import array as ary; from numpy import log as ln
from numpy import cos, sin, pi, sqrt, exp, arccos;
tau = 2*pi
import numpy as np;
from matplotlib import pyplot as plt
import pandas as pd
import glob
dirs = glob.glob('*/')

def number_of_zeros_on_the_left(series):
    tf_string = ''.join([str(b)[0] for b in series.values==0])
    return len(tf_string) - len(tf_string.lstrip('T'))

if __name__=='__main__':
    reactions = []
    R = pd.read_csv(dirs[0]+'R.csv', index_col=[0])
    initial_response, final_response = R[R['0']==-1], R[R['0']==-1] # created empty dataframe to append rows to 
    for d in dirs:
        thick = pd.read_csv(d+'thicknesses.csv', index_col=[0])
        R = pd.read_csv(d+'R.csv', index_col=[0])

        #filter away the irrelevant lines
        mask = np.logical_and(0.01<thick['max thickness'], thick['min thickness']<0.01) # condition for filtering
        thick = thick[mask]
        for rname, tf in mask.items():
            if tf:
                try:
                    initial_response = initial_response.append(R.loc[rname], verify_integrity=True)
                except ValueError: # due to having duplicate lines 
                    assert all(initial_response.loc[rname] == R.loc[rname]), "The existing response in the dataframe must match the new response!"
                    # print(rname, 'already present in initial_response')
        for name, line in thick.iterrows():
            reactions.append(line)
    #Lets assume we're all using 0.1 mm thick foil.
    # response matrix row [k] = R.csv.loc[k]  * 0.01 * rr.csv['mole (parent) per cm thickness'].loc[k]
    # i.e. The number of transmutation per mole parent of nuclide [k] per cm^-2 of neutron fluence (in the i^th bin) * 0.01 mole per 0.1 cm foil = number of transmutation per foil
    reaction_mole_parent_per_foil = {i.name:i['mole per cm thickness']*0.01 for i in reactions}
    
    order = []
    for rname, response in initial_response.iterrows():
        final_response = final_response.append(response * reaction_mole_parent_per_foil[rname], verify_integrity=True)
        order.append(number_of_zeros_on_the_left(response))
    
    new_indices = ary(final_response.index)[np.argsort(order)]
    final_response = final_response.reindex(new_indices)
    final_response.to_csv('final_response.csv', index_label='rname')