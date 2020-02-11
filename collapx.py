from numpy import exp, cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from numpy import log as ln
import scipy
from scipy.stats.mstats import gmean
from openmc.data.function import INTERPOLATION_SCHEME #y,x
import openmc
import warnings
import pandas as pd
from ReadData import read_reformatted, Reaction # the Reaction class is needed in order to load the pkl file in without raising an error.
import os, sys
import pickle
from convert_flux import get_scheme, loglog, histogramic, Integrate
from convert_flux import get_continuous_flux, get_gs_ary, integrate_continuous_flux # will have to use the integrate_continuous_flux soon

ALLOW_UNSTABLE_PARENT = True #allowing the unstable nuclide e.g. C14.
#Set this to true most of the time. There is another option at the top of convert2R to filter out non-naturally occurring elements.
NEGLIGIBLE_FLUX_ABOVE = 1.195E7 # If the cross-section data does not extend up to this value, then we have a problem.
THERMAL_E = 1/40 #eV
VERBOSE = False # print all of the ignored nuclide
FOCUS_MODE = True
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

def save_rnr(rnr, working_dir):
    loc = os.path.join(working_dir, 'reaction_and_radiation.pkl')
    with open(loc, 'wb') as f:
        pickle.dump(rnr, f)
    return

def read_rnr(working_dir):
    loc = os.path.join(working_dir, 'reaction_and_radiation.pkl')
    with open(loc, 'rb') as f:
        rnr = pickle.load(f)
    return rnr

class ReactionAndRadiation(): # for storing collapsed cross-sections, specific to that group structure (and apriori, if a non-histogramic apriori is used.).
    def __init__(self, sigma, decay_products_and_radiation, parent_nuclide, thermal_xs):
        self.sigma = sigma
        self.decay = decay_products_and_radiation
        self.parent= parent_nuclide
        self.thermal_xs = thermal_xs

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

# def interpolate_flux(flux_per_eV, flux_gs, scheme=histogramic):
#     '''
#     By default, assume the flux is flat, i.e. intrabin distribution is independent of interbin differences.
#     i.e. histogramic distribution.
#     Obviously this will produce discontinuities, but
#     1. There is no need for flux to be continuous
#     2. Once the cross-sections are folded, and flux starts to vary, other interpolations will become discontinuous too.
#     3. Flat distribution is the only interpolation method where the shape remains unchanged as the total flux in bin varies,
#         i.e. it is independent of interbin flux variation.
#     '''
#     assert all(np.diff(flux_gs.flatten())[1::2]==0), "The lower bounds of the energy groups must equal the upper bound of the previous group in order to interpolate properly."
#     group_bounds = np.hstack([ary(flux_gs)[:,0], flux_gs[-1][-1]])
#     extended_flux= np.hstack([flux_per_eV, flux_per_eV[-1]])
#     flux_func = openmc.data.Tabulated1D(group_bounds, extended_flux, breakpoints=[len(extended_flux)], interpolation = [scheme])
#     return flux_func

def scipy_integrate(func, a, b, error_msg, rtol=1E-3):
    with warnings.catch_warnings(record=True) as w:
        integral, error = scipy.integrate.quadrature(func, a, b, rtol=rtol)
        if w:
            print(str(w[0].message), "for ", error_msg)
    return integral, error

def collap_xs(sigma, gs_ary, error_msg, apriori_per_eV_func=None): # apriori should be given in per MeV
    if max(sigma.x)<NEGLIGIBLE_FLUX_ABOVE: # don't bother if the record is incomplete, energy-range wise.
        return None
    
    sigma_g = []
    if type(apriori_per_eV_func)==type(None):
        apriori_per_eV_func = integrate_continuous_flux( np.ones(len(gs_ary)), gs_ary )
    use_analytical_integration = True
    if use_analytical_integration:
        I = Integrate(sigma)
        for i in range(len(gs_ary)):
            numinator = I(*gs_ary[i]) #**Change this in the future!
            denominator = np.diff(gs_ary[i])
            sigma_g.append(numinator/denominator)
        return ary(sigma_g).flatten().tolist()
    else:
        for i in range(len(gs_ary)):
            integrand = lambda x : apriori_per_eV_func(x) * sigma(x)
            numinator = scipy_integrate(integrand, gs_ary[i][0], gs_ary[i][1], error_msg='folding flux '+error_msg+' bin '+str(i)+" within energy range of"+ str(gs_ary[i])+" eV")
            denominator = scipy_integrate(apriori_per_eV_func, gs_ary[i][0], gs_ary[i][1],'normalization factor for '+error_msg+' bin '+ str(i))
            sigma_g.append( numinator[0]/denominator[0] ) # index[0] for choosing the centroid instead of the uncertainty value.
            # Change ^ this line if you want implement covariance in the future.
        return sigma_g #unit: barns

# def read_apriori_and_gs_df(out_dir, apriori_in_fmt='integrated', apriori_gs_file=None, apriori_multiplier=1, gs_multipliers=1):
#     try:
#         import os, sys
#         assert os.path.exists(out_dir), "Please create directory {0} to save the output files in first.".format(out_dir)
#         gs_file = os.path.join(out_dir, "gs.csv")
#         apriori_file = os.path.join(out_dir, "apriori.csv")
#         print("scaling up/down the group structure's numerical value by {0} to obtain the group structure in eV".format(gs_multipliers))
#         gs_array = pd.read_csv( gs_file, sep=',').values *gs_multipliers
#         print("Reading in the a priori file from {0} in the format of {1} flux, by scaling its numerical values up/down by a factor of{2}".format(apriori_file, apriori_in_fmt, apriori_multiplier))
#         apriori_df = pd.read_csv( apriori_file, sep=',|±', engine='python')*apriori_multiplier # tell the apriori and the uncertainty.
#         if type(apriori_gs_file)==type(None):
#             apriori_gs = gs_array
#         else:
#             raise DeprecationWarning(f"Flux rebinning should be done in a different module than {sys.argv[0]}, e.g. in convert_flux.py")
#             sys.exit()
#             #later on, can implement rebinner inside here to change the group structure when necessary.
#         apriori_per_eV_df = flux_conversion(flux = apriori_df.T, gs_in_eV = apriori_gs, in_fmt = apriori_in_fmt, out_fmt = 'per eV')
#     except Exception as e:
#         print("{0} occurred when calling main_collapse(gs_file, apriori_file),".format(repr(e)))
#         print("Possibly due to incorrect file format/non-existent files.")
#         print("Make sure that they are in {0}, {1}".format(gs_file, apriori_file))
#         print("apriori_file should be in 2 columns, with the headers 'value' and 'uncertainies',")
#         print("gs_file should be in 2 columns, with the headers 'min' and 'max' denoting the energy boundaries of each bin,")
#         print("    (For a continuous spectrum, the max of this bin should be the same as min of the next bin)")
#         import sys
#         sys.exit()
#     return apriori_per_eV_df.T, gs_array

def main_collapse(apriori_func, gs_array, rdict, dec_r):
    void_reactions = [] # these are the records which are voided because they are 
    reaction_and_radiation = {}
    print("Collapsing cross-sections using gaussian quadrature...")
    for iso, mts in rdict.items():
        for mt, r in mts.items():
            rname = iso+"-"+str(mt)
            if r.products_name: #only care about isotopes that has a product name recorded.
                #
                if all([i in dec_r.keys() for i in r.products_name]): #only plot it if we know all of the decay products.
                    if ALLOW_UNSTABLE_PARENT or r.parent['stable']:
                        if not FOCUS_MODE:
                            print("collapsing", rname)
                        sigma_g = collap_xs(r.sigma, gs_array, rname, apriori_per_eV_func=apriori_func) #apriori_and_unc.values[:,0])
                        thermal_xs = r.sigma([THERMAL_E])[0]
                        if type(sigma_g)==type(None):
                            void_reactions.append(rname)
                        else:
                            reaction_and_radiation[rname] = ReactionAndRadiation(sigma_g, [dec_r[i] for i in sorted(set(r.products_name))], r.parent, thermal_xs)
                    elif VERBOSE:
                        print("Ignoring ", rnmae, "as the parent is unstable.")
                    # with open("output/"+file_name, "w") as f:
                    #     f.write(np.array2string(sigma_g))
                elif 0<sum([i in dec_r.keys() for i in r.products_name])< len(r.products_name):
                    print("We're getting ",rname,"which has", r.products_name, "but has incomplete record")
                    print([i in dec_r.keys() for i in r.products_name])
                    pass
                elif VERBOSE:
                    print("Ignoring ", rname, "as the daughter nuclide's decay record does not exist.")
                    # pass
            else:
                print(rname, "Does not have reaction products listed")
    if VERBOSE:
        print("Ignored the reactions nuclide due to incomplete cross-section data coverage over the "
                +"interested range of flux (E>{0}eV) :".format(NEGLIGIBLE_FLUX_ABOVE), void_reactions)
    print("found {0} reactions which has recorded cross-section and products.".format(len(reaction_and_radiation)))
    return reaction_and_radiation

def slideshow(rdict, gs_array, word='', min_xs=0, min_xs_E=8E6):
    for gnd_name, mts in rdict.items():
        for mt, r in mts.items():
            if word in gnd_name:
                if r.sigma(min_xs_E)>min_xs:
                    plt.title(gnd_name+"-"+str(mt)+"-"+all_mts[mt])
                    plt.loglog(r.sigma.x, r.sigma.y)
                    plt.xticks(gs_array.flatten())
                    plt.xlim(*(gs_array.flatten()[[0,-1]]))
                    plt.xticks(np.hstack([gs_array.flatten()[[0,-1]],min_xs_E]))
                    plt.draw()
                    plt.pause(0.35)
                    plt.clf()

if __name__=='__main__':
    rdict, dec_r, all_mts = read_reformatted(sys.argv[-1])
    apriori_func = get_continuous_flux(sys.argv[-1])
    gs_array = get_gs_ary(sys.argv[-1])
    rnr = main_collapse(apriori_func, gs_array, rdict, dec_r) # put them all into the same object for easy computation in the next step.
    save_rnr(rnr, sys.argv[-1])