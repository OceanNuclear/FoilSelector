import numpy as np
import uncertainties
import pandas as pd
import os, sys
import openmc
from numpy import sqrt
from numpy import log as ln
from tqdm import tqdm
MeV = 1E6
# debugging tools
sdir = lambda x: [i for i in dir(x) if '__' not in i]
import matplotlib.pyplot as plt
plot_tab = lambda tab: plt.plot(*[getattr(tab, ax) for ax in 'xy'])
haskey = lambda dict_instance, key: key in dict_instance.keys()

from openmc.data.reaction import REACTION_NAME
from openmc.data.endf import SUM_RULES

def exp(numbers):
    if isinstance(numbers, uncertainties.core.AffineScalarFunc):
        return uncertainties.unumpy.exp(numbers)
    else:
        return np.exp(numbers)

def welcome_message():
    try:
        folder_list = sys.argv[1:]
        assert len(folder_list)>0
        #Please update your python to 3.6 or later.
        print(f"Reading from {len(folder_list)} folders,")
        endf_file_list = []
        for folder in folder_list:
            endf_file_list += [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.endf') or file.endswith('.asc')]
    except (IndexError, AssertionError):
        print("usage:")
        print("'python "+sys.argv[0]+" folders/ containing/ endf/ files/ in/ descending/ order/ of/ priority/ [output/]'")
        print("where the outputs-saving directory 'output/' is only requried when read_apriori_and_gs_df is used.")
        print("Use wildcard (*) to replace directory as .../*/ if necessary")
        print("The endf files in question can be downloaded from online sources")
        # by running the make file.")
        # from https://www.oecd-nea.org/dbforms/data/eva/evatapes/eaf_2010/ and https://www-nds.iaea.org/IRDFF/
        # Currently does not support reading h5 files yet, because openmc does not support reading ace/converting endf into hdf5/reading endf using NJOY yet
        sys.exit()

    print(f"Found {len(endf_file_list)} regular files ending in '.endf' or '.asc'. Reading them ...")
    #read in each file:
    endf_data = []
    for path in tqdm(endf_file_list):
        try:
            endf_data += openmc.data.get_evaluations(path)#works with IRDFF/IRDFFII.endf and EAF/*.endf
        except:
            endf_data += [openmc.data.Evaluation(path),] # works with decay/decay_2012/*.endf
    return endf_data

def HPGe_efficiency_curve_generator(working_dir, deg=4, cov=True):
    '''
    according to Knoll (equation 12.32), polynomial fit in log-log space is the best.
    This program is trying to do the same.
    This polynomial in log-log space fit is confirmed by
    @article{kis1998comparison,
    title={Comparison of efficiency functions for Ge gamma-ray detectors in a wide energy range},
    author={Kis, Zs and Fazekas, B and {\"O}st{\"o}r, J and R{\'e}vay, Zs and Belgya, T and Moln{\'a}r, GL and Koltay, L},
    journal={Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment},
    volume={418},
    number={2-3},
    pages={374--386},
    year={1998},
    publisher={Elsevier}
    }
    '''
    import glob
    file_location = glob.glob(os.path.join(working_dir, '*photopeak_efficiency*.csv'))[0]
    datapoints = pd.read_csv(file_location)
    assert "energ" in datapoints.columns[0].lower(), "The file must contain a header. Energy (eV/MeV) has to be placed in the first column"
    E, eff = datapoints.values.T[:2]
    if 'MeV' in datapoints.columns[0] or 'MeV' in file_location:
        E = MeV*E
    if datapoints.values.T[2:].shape[0]>0: # assume that column is the error
        sigma = datapoints.values.T[2]
        print("Using the 3rd column of ", file_location, r" as error on the efficiency measurements (\sigma)")
        cov = 'unscaled'
        w = 1/sigma**2
    else:
        w = None
    from scipy import polyfit
    p = polyfit(ln(E), ln(eff), deg, w=w, cov=cov) #make sure that the curve points downwards at high energy 
    if cov:
        p, pcov = p
    #iteratively try increasing degree of polynomial fits.
    #Currently the termination condition is via making sure that the curve points downwards at higher energy (leading coefficient is negative).
    #However, I intend to change the termination condition to Bayesian Information Criterion instead.
    if not p[0]<0:
        mindeg, maxdeg = 2,6
        for i in range(mindeg, maxdeg+1):
            p = polyfit(ln(E), ln(eff), deg, w=w, cov=cov)            
            if cov:
                p, pcov = p
            if p[0]<0:
                print("a {0} order polynomial fit to the log(energy)-log(efficiency) curve is used instead of the default {1} order".format(i, deg))
                break
            elif i==maxdeg:#if we've reached the max degree tested and still haven't broken the loop, it means none of them fits.
                print("None of the polynomial fit in log-log space is found to extrapolate properly! using the default {0} order fit...".format(deg) )
                p = polyfit(ln(E), ln(eff), deg)
    print("The covariance matrix is", pcov)

    def efficiency_curve(E):
        if isinstance(E, uncertainties.core.AffineScalarFunc):
            lnE = ln(E.n)
        else:
            lnE = ln(E)
        lneff = np.sum( [p[::-1][i]* lnE**i for i in range(len(p))], axis=0) #coefficient_i * x ** i
        
        if cov:
            lnE_powvector = [lnE**i for i in range(len(p))][::-1]
            variance_on_lneff = (lnE_powvector @ pcov @ lnE_powvector) # variance on lneff
            if isinstance(E, uncertainties.core.AffineScalarFunc):
                error_of_lnE = E.s/E.n
                variance_from_E = sum([p[::-1][i]*i*lnE**(i-1) for i in range(1, len(p))])**2 * (error_of_lnE)**2
                variance_on_lneff += variance_from_E
            lneff_variance = exp(lneff)**2 * variance_on_lneff
            return uncertainties.core.Variable( exp(lneff), sqrt(lneff_variance) )
        else:
            return exp(lneff)
    return efficiency_curve


MT_to_nuc_num = {
    2:(0, 0),
    4:(0, 0),
    11:(-1, -3),
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
for i in range(51, 92): # (z,n?)
    MT_to_nuc_num[i] = tuple([*MT_to_nuc_num[2], i-50])
for i in range(600, 649): # (z, p?)
    MT_to_nuc_num[i] = tuple([*MT_to_nuc_num[103], i-600])
MT_to_nuc_num[649] = MT_to_nuc_num[103]
for i in range(650, 699): # (z, d?)
    MT_to_nuc_num[i] = tuple([*MT_to_nuc_num[104], i-650])
MT_to_nuc_num[699] = MT_to_nuc_num[104]
for i in range(700, 749): # (z, t?)
    MT_to_nuc_num[i] = tuple([*MT_to_nuc_num[105], i-700])
MT_to_nuc_num[749] = MT_to_nuc_num[105]
for i in range(750, 799): # (z, He3?)
    MT_to_nuc_num[i] = tuple([*MT_to_nuc_num[106], i-750])
MT_to_nuc_num[799] = MT_to_nuc_num[106]
for i in range(800, 849):
    MT_to_nuc_num[i] = tuple([*MT_to_nuc_num[107], i-800])
MT_to_nuc_num[849] = MT_to_nuc_num[107]
for i in range(875, 891):
    MT_to_nuc_num[i] = tuple([*MT_to_nuc_num[16], i-875])
MT_to_nuc_num[891] = MT_to_nuc_num[16]