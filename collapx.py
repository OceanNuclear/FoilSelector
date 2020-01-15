from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from ReadData import *
import scipy
import openmc
from openmc.data.function import INTERPOLATION_SCHEME
import time 
from numpy import array as ary; import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean
from math import fsum
from numpy import log as ln
MeV = 1E6 #multiply by this to convert values in MeV to eV
def val_to_key_lookup(dic, val):
    for k,v in dic.items():
        if v==val:
            return k
scheme = val_to_key_lookup(INTERPOLATION_SCHEME, 'log-log')

class ReactionAndRadiation():
    def __init__(self, sigma, decay_products_and_radiation):
        self.sigma = sigma
        self.decay = decay_products_and_radiation

def flux_conversion(flux, gs_in_eV, in_fmt, out_fmt):
    if in_fmt=='per MeV':
        flux_per_eV = flux/MeV
    elif in_fmt=='integrated':
        flux_per_eV = flux/np.diff(gs_in_eV, axis=1).flatten()
    elif in_fmt=='PUL':
        leth_space = np.diff(ln(gs_in_eV), axis=1).flatten()
        flux_integrated = flux*leth_space
        flux_per_eV = flux_conversion(flux_integrated, gs_in_eV, 'integrated', flux_per_eV)
    else:
        assert in_fmt=="per eV", "the input format 'i' must be one of the following 3='integrated'|'PUL'(per unit lethargy)|'per (M)eV'"
        flux_per_eV = flux

    if out_fmt=='per MeV':
        return flux_per_eV*MeV
    elif out_fmt=='integrated':
        return flux_per_eV * np.diff(gs_in_eV, axis=1).flatten()
    elif out_fmt=='PUL':
        leth_space = np.diff(ln(gs_in_eV), axis=1).flatten()
        flux_integrated = (flux_per_eV, gs_in_eV,'per eV', 'integrated')
        return flux_integrated/leth_space
    else:
        assert out_fmt=='per eV', "the input format 'i' must be one of the following 3='integrated'|'PUL'(per unit lethargy)|'per (M)eV'"
        return flux_per_eV

def collap_xs(sigma, gs, apriori_per_eV=None): # apriori should be given in per MeV
    RESOLUTION = 100
    '''
    if implicit and type(apriori_per_eV)==type(None):
        try:
            with warnings.catch_warnings(record=True) as w:
                sigma_int = openmc.data.Tabulated1D(sigma.x, sigma.integral(), sigma.breakpoints)
                sigma_int_val = [ (sigma_int(max(bounds)) - sigma_int(min(bounds))) for bounds in gs]
                assert len(w)==0, "if an error is raised, a NaN will be produced and the result will be unusable."
                return sigma_int_val/np.diff(gs, axis=1).flatten()
        except:
            pass
    '''
    sigma_g = []
    centroid = [gmean(bounds) for bounds in gs]
    if type(apriori_per_eV)==type(None):
        apriori_per_eV = np.ones(len(centroid))
    extended_gs = np.concatenate([[gs[0][0]], centroid, [gs[-1][-1]]])
    exnteded_ap = np.concatenate([[apriori_per_eV[0]], apriori_per_eV, [apriori_per_eV[-1]]])
    apriori_func = openmc.data.Tabulated1D(extended_gs, exnteded_ap, interpolation = scheme)
    for i in range(len(centroid)):
        integrand = lambda x : apriori_func(x) * sigma(x)
        numinator = scipy.integrate.quadrature(integrand, gs[i][0], gs[i][1], rtol=1E-2)
        denominator = scipy.integrate.quadrature(apriori_func, gs[i][0], gs[i][1], rtol=1E-2) # It's not my job to find and use a good integrator; It's FISPACT's job.
        sigma_g.append( sum(numinator)/sum(denominator) )
    return sigma_g #unit: barns

apriori_and_unc = pd.read_csv('output/apriori.csv', sep=',|±', engine='python')*MeV # tell the apriori and the uncertainty.
gs = (pd.read_csv('output/gs.csv', sep=',')*MeV).values
apriori_per_eV = flux_conversion(flux = apriori_and_unc['value'].values, gs_in_eV = gs, in_fmt = 'integrated', out_fmt = 'per MeV')

reaction_and_radiation = {}


print("Collapsing cross-sections using gaussian quadrature...")
for iso, mts in rdict.items():
    for mt, r in mts.items():
        rname = iso+"-"+str(mt)
        if r.products_name:
            '''
            try:
                daugh_nuc_record = openmc.data.gnd_name(r.products_name[0]//1000, r.products_name[0]%1000)
                paren_nuc = openmc.data.zam(iso)[:2]
                diff = ary(translation[mt])
            except:
                pass
            daugh_nuc = openmc.data.gnd_name( *(ary(paren_nuc) + diff) ) #calculated iso
            if daugh_nuc!=daugh_nuc_record:
                print(f"MT={mt}, product does not match expectation: {daugh_nuc} expected, {daugh_nuc_record} found.")
            else:
                print(f"MT={mt}, matches")
            '''
            if all([i in dec_r.keys() for i in r.products_name]): #only plot it if we know all of the decay products.
                sigma_g = collap_xs(r.sigma, gs, apriori_per_eV = apriori_per_eV) #apriori_and_unc.values[:,0])
                reaction_and_radiation[rname] = ReactionAndRadiation(sigma_g, [dec_r[i] for i in r.products_name])
                # with open("output/"+file_name, "w") as f:
                #     f.write(np.array2string(sigma_g))
        # else:
        #     print(rname, "Does not have reaction products listed")
print(f"found {len(reaction_and_radiation)} reactions which has recorded cross-section and products.")

