from numpy import exp, cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from numpy import log as ln
import scipy
from scipy.stats.mstats import gmean
from openmc.data.function import INTERPOLATION_SCHEME
import openmc
import warnings
import pandas as pd
from ReadData import main_read
from collections.abc import Iterable
from math import fsum
import os, sys

ALLOW_UNSTABLE_PARENT = True #allowing the unstable nuclide e.g. C14.
#Set this to true most of the time. There is another option at the top of convert2R to filter out non-naturally occurring elements.
VERBOSE = False # print all of the ignored nuclide
NEGLIGIBLE_FLUX_ABOVE = 1E7 # If the cross-section data does not extend up to this value, then we have a problem:
    #Because the nuclear data will be 
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
histogramic = get_scheme('histogram')

def area_between_2_pts(xy1,xy2, xi,scheme):
    x1, y1, x2, y2, xi = np.hstack([xy1, xy2, xi]).flatten()
    assert x1<=xi<=x2, "xi must be between x1 and x2"
    x_ = xi-x1
    if x1==x2:
        return 0.0  #catch all cases with zero-size width bins
    if y1==y2 or scheme==1:# histogramic/ flat interpolation
        return y1*x_ #will cause problems in all log(y) interpolation schemes if not caught
    
    dy, dx = y2-y1, x2-x1
    logy, logx = [bool(int(i)) for i in bin(scheme-2)[2:].zfill(2)]
    if logx:
        assert(all(ary([x1,x2,xi])>0)), "Must use non-zero x values for these interpolation schemes"
        lnx1, lnx2, lnxi = ln(x1), ln(x2), ln(xi)
        dlnx = lnx2 - lnx1
    if logy:
        assert(all(ary([y1,y2])>0)), "Must use non-zero y values for these interpolation schemes"
        lny1, lny2 = ln(y1), ln(y2)
        dlny = lny2 - lny1

    if scheme==2: # linx, liny
        m = dy/dx
        if xi==x2:
            return dy*dx/2 + y1*dx
        else:
            return y1*x_ + m*x_**2 /2
    if scheme==3: # logx, liny
        m = dy/dlnx
        if xi==x2:
            return y1*dx + m*(x2*dlnx-dx)
        else:
            return y1*x_ + m*(xi*(lnxi-lnx1)- x_)
            return (y1 - m*lnx1)*x_ + m*(-x_+xi*lnxi-x1*lnx1)
    if scheme==4: # linx, logy
        m = dlny/dx
        if xi==x2:
            return 1/m *dy
        else:
            return 1/m *y1*(exp(x_*m)-1)
    if scheme==5:
        m = dlny/dlnx
        if m==-1:
            return y1 * x1 * (lnxi-lnx1)
        if xi==x2:
            return y1/(m+1) * ( x2 * (x2/x1)**m - x1 )
        else:
            return y1/(m+1) * ( xi * (1 + x_/x1)**m - x1 )
    else:
        raise AssertionError("a wrong interpolation scheme {0} is provided".format(scheme))

class Integrate():
    def __init__(self, sigma):
        self.x = sigma.x
        self.y = sigma.y
        self.xy = ary([self.x, self.y])
        assert all(np.diff(self.x)>=0), "the x axis must be inputted in ascending order"
        self.interpolations = []
        self.next_area = []
        for i in range(len(self.x)-1):
            break_region = np.searchsorted(sigma.breakpoints, i+1, 'right')
            self.interpolations.append(sigma.interpolation[break_region])
            # region_num = np.searchsorted(i, self.breakpoints, 'right')
            # scheme = self.interpolation[region_num]
            x1, x2 = sigma.x[i:i+2]
            xy1 = ary([x1, sigma.y[i]])
            xy2 = ary([x2, sigma.y[i+1]])
            self.next_area.append(area_between_2_pts(xy1, xy2, x2, scheme=self.interpolations[i] ))
    def __call__(self, a, b):
        assert np.shape(a)==np.shape(b), "There must be as many starting points as the ending points to the integrations."
        if isinstance(a, Iterable):
            return ary([self(ai, bi) for (ai,bi) in zip(a,b)]) #catch all iterables cases.
            # BTW, don't ever input a scalar into __call__ for an openmc.data.Tabulated1D function.
            # It's going to return a different result than if you inputted an Iterable.
            # Integrate(), on the other hand, doesn't have this bug.
        
        a, b = np.clip([a,b], min(self.x), max(self.x)) # assume if no data is recorded at the low energy limit.
        ida, idb = self.get_region(a), self.get_region(b)
        total_area = self.next_area[ida:idb]
        total_area.append(-area_between_2_pts(self.xy[:,ida], self.xy[:,ida+1], a, self.interpolations[ida]))
        total_area.append(area_between_2_pts(self.xy[:,idb], self.xy[:,idb+1], b, self.interpolations[idb]))
        return fsum(total_area)

    def get_region(self, x):
        idx = np.searchsorted(self.x, x, 'right')-1
        assert all([x<=max(self.x)]),"Out of bounds of recorded x! (too high)"
        # assert all([0<=idx]), "Out of bounds of recorded x! (too low)" #doesn't matter, simply assume zero. See __call__
        idx -= x==max(self.x) * 1 # if it equals eactly the upper limit, we need to minus one in order to stop it from overflowing.
        return np.clip(idx, 0, None)

class ReactionAndRadiation(): # for storing collapsed cross-sections, specific to that group structure (and apriori, if a non-histogramic apriori is used.).
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

def interpolate_flux(flux_per_eV, flux_gs, scheme=histogramic):
    '''
    By default, assume the flux is flat, i.e. intrabin distribution is independent of interbin differences.
    i.e. histogramic distribution.
    Obviously this will produce discontinuities, but
    1. There is no need for flux to be continuous
    2. Once the cross-sections are folded, and flux starts to vary, other interpolations will become discontinuous too.
    3. Flat distribution is the only interpolation method where the shape remains unchanged as the total flux in bin varies,
        i.e. it is independent of interbin flux variation.
    '''
    assert all(np.diff(flux_gs.flatten())[1::2]==0), "The lower bounds of the energy groups must equal the upper bound of the previous group in order to interpolate properly."
    group_bounds = np.hstack([ary(flux_gs)[:,0], flux_gs[-1][-1]])
    extended_flux= np.hstack([flux_per_eV, flux_per_eV[-1]])
    flux_func = openmc.data.Tabulated1D(group_bounds, extended_flux, breakpoints=[len(extended_flux)], interpolation = [scheme])
    return flux_func

def scipy_integrate(func, a, b, error_msg, rtol=1E-3):
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
    #Can interpolate sigma such that its lowest value extends to E=0, and max E value extends to E=max(gs.flatten())
    if max(sigma.x)<NEGLIGIBLE_FLUX_ABOVE: # don't bother if the record is incomplete, energy-range wise.
        return None
    
    sigma_g = []
    if type(apriori_per_eV_func)==type(None):
        apriori_per_eV_func = interpolate_flux( np.ones(len(gs_ary)), gs_ary )
    if True:
        I = Integrate(sigma)
        for i in range(len(gs_ary)):
            numinator = I(*gs_ary[i])
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

def read_apriori_and_gs_df(out_dir=sys.argv[-1], apriori_in_fmt='integrated', apriori_gs_file=None, apriori_multiplier=1, gs_multipliers=1):
    try:
        assert os.path.exists(out_dir), "Please create directory {0} to save the output files in first.".format(out_dir)
        gs_file = os.path.join(out_dir, "gs.csv")
        apriori_file = os.path.join(out_dir, "apriori.csv")
        print("scaling up/down the group structure's numerical value by {0} to obtain the group structure in eV".format(gs_multipliers))
        gs_array = pd.read_csv( gs_file, sep=',').values *gs_multipliers
        print("Reading in the a priori file from {0} in the format of {1} flux, by scaling its numerical values up/down by a factor of{2}".format(apriori_file, apriori_in_fmt, apriori_multiplier))
        apriori_df = pd.read_csv( apriori_file, sep=',|±', engine='python')*apriori_multiplier # tell the apriori and the uncertainty.
        if type(apriori_gs_file)==type(None):
            apriori_gs = gs_array
        else:
            raise Warning("Flux rebinning is not yet implemented!")
            #later on, can implement rebinner inside here to change the group structure when necessary.
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
                        print("collapsing", rname)
                        sigma_g = collap_xs(r.sigma, gs_array, rname, apriori_per_eV_func=apriori_func) #apriori_and_unc.values[:,0])
                        if type(sigma_g)==type(None):
                            void_reactions.append(rname)
                        else:
                            reaction_and_radiation[rname] = ReactionAndRadiation(sigma_g, [dec_r[i] for i in sorted(set(r.products_name))], r.parent)
                    elif VERBOSE:
                        print("Ignoring ", rnmae, "as the parent is unstable.")
                    # with open("output/"+file_name, "w") as f:
                    #     f.write(np.array2string(sigma_g))
                elif VERBOSE:
                    print("Ignoring ", rname, "as the daughter nuclide's decay record is incomplete.")
            elif VERBOSE:
                print(rname, "Does not have reaction products listed")
    if VERBOSE:
        print("Ignored the reactions nuclide due to incomplete cross-section data coverage over the "
                +"interested range of flux (E>{0}eV) :".format(NEGLIGIBLE_FLUX_ABOVE), void_reactions)
    print("found {0} reactions which has recorded cross-section and products.".format(len(reaction_and_radiation)))
    return reaction_and_radiation

def rebin(flux, old_gs, new_gs):
    return flux

def slideshow(rdict, word=''):
    for gnd_name, mts in rdict.items():
        for mt, r in mts.items():
            if word in gnd_name:
                plt.title(gnd_name+"-"+str(mt)+"-"+all_mts[mt])
                plt.loglog(r.sigma.x, r.sigma.y)
                plt.xticks(gs_array.flatten())
                plt.xlim(*(gs_array.flatten()[[0,-1]]))
                plt.draw()
                plt.pause(0.35)
                plt.clf()

if __name__=='__main__':
    apriori_and_unc, gs_array = read_apriori_and_gs_df(apriori_multiplier=1E5, gs_multipliers=MeV) # apriori is read in as eV^-1
    apriori_func = interpolate_flux(apriori_and_unc['value'].values, gs_array) # Can use a different flux profile if you would like to.
    rdict, dec_r, all_mts = main_read()
    reaction_and_radiation = main_collapse(apriori_func, gs_array, rdict, dec_r)