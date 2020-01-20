from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from numpy import log as ln
from scipy.stats.mstats import gmean
from openmc.data.function import INTERPOLATION_SCHEME
import openmc
import warnings
import pandas as pd
from ReadData import main_read

ALLOW_UNSTABLE_PARENT = True #allowing the unstable nuclide e.g. C14.
#Set this to true most of the time. There is another option at the top of convert2R to filter out non-naturally occurring elements.
VERBOSE = False # print all of the ignored nuclide
WARNING_THRESHOLD=0.01
'''
Purpose:
    Take in the variable rdict,
    read in the apriori, group structure + flux shape
    collapse the sigma according to the group structure, by weighting using the flux.
        such that when the integrated flux is mulitplied onto sigma_g, we still get a correct reaction rate(assuming the flux is still of the same shape as the apriori).
    and output a dictionary of reaction_and_radiation objects
Exclusions:
    Only files with corresponding decay data will be parsed onto the next stage, inside the reaction_and_radiation object.
'''

MeV = 1E6 #multiply by this to convert values in MeV to eV

def val_to_key_lookup(dic, val):
    for k,v in dic.items():
        if v==val:
            return k
def get_scheme(scheme):
    return val_to_key_lookup(INTERPOLATION_SCHEME, scheme)
loglog = get_scheme('log-log')

class ReactionAndRadiation():
    def __init__(self, sigma, decay_products_and_radiation, parent_nuclide):
        self.sigma = sigma
        self.decay = decay_products_and_radiation
        self.parent= parent_nuclide

def flux_conversion(flux, gs_in_eV, in_fmt, out_fmt):
    if in_fmt=='per MeV':
        flux_per_eV = flux/MeV
    elif in_fmt=='integrated':
        flux_per_eV = flux/np.diff(gs_in_eV, axis=1).flatten()
    elif in_fmt=='PUL':
        leth_space = np.diff(ln(gs_in_eV), axis=1).flatten()
        flux_integrated = flux*leth_space
        flux_per_eV = flux_conversion(flux_integrated, gs_in_eV, 'integrated', 'per eV')
    else:
        assert in_fmt=="per eV", "the input format 'i' must be one of the following 4='integrated'|'PUL'(per unit lethargy)|'per (M)eV'"
        flux_per_eV = flux

    if out_fmt=='per MeV':
        return flux_per_eV*MeV
    elif out_fmt=='integrated':
        return flux_per_eV * np.diff(gs_in_eV, axis=1).flatten()
    elif out_fmt=='PUL':
        leth_space = np.diff(ln(gs_in_eV), axis=1).flatten()
        flux_integrated = flux_conversion(flux_per_eV, gs_in_eV,'per eV', 'integrated')
        return flux_integrated/leth_space
    else:
        assert out_fmt=='per eV', "the input format 'i' must be one of the following 4='integrated'|'PUL'(per unit lethargy)|'per (M)eV'"
        return flux_per_eV

def interpolate_flux(flux_per_eV, flux_gs, scheme=loglog):
    centroid = [gmean(bounds) for bounds in flux_gs]
    extended_gs = np.concatenate([[flux_gs[0][0]], centroid, [flux_gs[-1][-1]]])
    extended_fl = np.concatenate([[flux_per_eV[0]], flux_per_eV, [flux_per_eV[-1]]])
    flux_func = openmc.data.Tabulated1D(extended_gs, extended_fl, interpolation = scheme)
    return flux_func

def integrate(func, a, b, error_msg, rtol=1E-2):
    with warnings.catch_warnings(record=True) as w:
        integral, error = scipy.integrate.quadrature(func, a, b, rtol=rtol)
        if w:
            print(str(w[0].message), "for ", error_msg)
    return integral, error

def collap_xs(sigma, gs_ary, error_msg, apriori_per_eV_func=None): # apriori should be given in per MeV
    '''
    if implicit and type(apriori_per_eV)==type(None):
        try:
            with warnings.catch_warnings(record=True) as w:
                sigma_int = openmc.data.Tabulated1D(sigma.x, sigma.integral(), sigma.breakpoints)
                sigma_int_val = [ (sigma_int(max(bounds)) - sigma_int(min(bounds))) for bounds in gs_ary]
                assert len(w)==0, "if an error is raised, a NaN will be produced and the result will be unusable."
                return sigma_int_val/np.diff(gs_ary, axis=1).flatten()
        except:
            pass
    '''
    sigma_g = []
    if type(apriori_per_eV_func)==type(None):
        apriori_per_eV_func = interpolation(np.ones(len(gs_ary)), gs_ary, scheme=get_scheme('linear-linear'))

    for i in range(len(gs_ary)):
        integrand = lambda x : apriori_per_eV_func(x) * sigma(x)
        # It's not my job to find and use a good integrator; It's FISPACT's job.
        # Therefore I will move all this duty to fispact later.
        numinator = integrate(integrand, gs_ary[i][0], gs_ary[i][1], error_msg='folding flux '+error_msg+' bin '+str(i)+" within energy range "+ str(gs_ary[i])+" eV")
        denominator = integrate(apriori_per_eV_func, gs_ary[i][0], gs_ary[i][1],'normalization factor for '+error_msg+' bin '+ str(i))
        if VERBOSE:
            if max(sigma.x)<max(gs_ary.flatten()) and (sigma.y[-1] > WARNING_THRESHOLD*max(sigma.y)):
                print("Warning: the sigma at high energy is non-zero, but is set to zero by force by openmc!")
                print("I.e. the nuclear data does not cover the entire range of energy required tobe sampled, but we are assuming that it is zero outisde of this range.")
        sigma_g.append( numinator[0]/denominator[0] ) # index[0] for choosing the centroid instead of the uncertainty value.
        # Change ^ this line if you want implement covariance in the future.
    return sigma_g #unit: barns

def read_apriori_and_gs_df(gs_file, apriori_file, apriori_in_fmt='integrated', apriori_gs_file=None, apriori_multiplier=1, gs_multipliers=1):
    try:
        gs_array = pd.read_csv( gs_file, sep=',').values *gs_multipliers
        apriori_df = pd.read_csv( apriori_file, sep=',|±', engine='python')*apriori_multiplier # tell the apriori and the uncertainty.
        if type(apriori_gs_file)==type(None):
            apriori_gs = gs_array
        apriori_per_eV_df = flux_conversion(flux = apriori_df.T, gs_in_eV = apriori_gs, in_fmt = apriori_in_fmt, out_fmt = 'per eV')
    except Exception as e:
        print("{0} occurred when calling main_collapse(gs_file, apriori_file),".format(repr(e)))
        print("Possibly due to incorrect file format/non-existent files.")
        print("Make sure that they are in {0}, {1}".format(gs_file, apriori_file))
        print("apriori_file should be in 2 columns, with the headers 'value' and 'uncertainies',")
        print("gs_file should be in 2 columns, with the headers 'min' and 'max' denoting the energy boundaries of each bin,")
        print("    (For a continuous spectrum, the max of this bin should be the same as min of the next bin)")
        import sys
        sys.exit()
    return apriori_per_eV_df.T, gs_array

def main_collapse(apriori_and_unc, gs_array, rdict, dec_r):
    apriori_func = interpolate_flux(apriori_and_unc['value'].values, gs_array) # Can use a different flux profile if you would like to.

    reaction_and_radiation = {}
    print("Collapsing cross-sections using gaussian quadrature...")
    for iso, mts in rdict.items():
        for mt, r in mts.items():
            rname = iso+"-"+str(mt)
            if r.products_name: #only care about isotopes that has a product name recorded.
                #
                if all([i in dec_r.keys() for i in r.products_name]): #only plot it if we know all of the decay products.
                    if ALLOW_UNSTABLE_PARENT or r.parent['stable']:
                        sigma_g = collap_xs(r.sigma, gs_array, rname, apriori_per_eV_func=apriori_func) #apriori_and_unc.values[:,0])
                        reaction_and_radiation[rname] = ReactionAndRadiation(sigma_g, [dec_r[i] for i in r.products_name], r.parent)
                    elif VERBOSE:
                        print("Ignoring ", rnmae, "as the parent is unstable.")
                    # with open("output/"+file_name, "w") as f:
                    #     f.write(np.array2string(sigma_g))
                elif VERBOSE:
                    print("Ignoring ", rname, "as the daughter nuclide's decay record is incomplete.")
            elif VERBOSE:
                print(rname, "Does not have reaction products listed")
    print("found {0} reactions which has recorded cross-section and products.".format(len(reaction_and_radiation)))
    return reaction_and_radiation

if __name__=='__main__':
    rdict, dec_r, all_mts = main_read()
    apriori_and_unc, gs_array = read_apriori_and_gs_df('output/gs.csv', 'output/apriori.csv', 'integrated', apriori_multiplier=1E5, gs_multipliers=MeV) # apriori is read in as eV^-1
    reaction_and_radiation = main_collapse(apriori_and_unc, gs_array, rdict, dec_r)