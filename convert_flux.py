from numpy import exp, cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from numpy import log as ln
from typing import Iterable
from math import fsum
from matplotlib import pyplot as plt
import os, glob, sys
from openmc.data import Tabulated1D
import pandas as pd
import openmc
from openmc.data.function import INTERPOLATION_SCHEME #y,x
import pickle
import csv
MeV=1E6
VERBOSE = True
# SHIFT = .7E6 # eV
SCALE_FACTOR = 1.04
RESOLUTION = 90
CHOICE = 1 # when running the 'Ohio/' folder, use CHOICE=0 for highest energy source (7 ish MeV); CHOICE=2 for lowest energy source.
def val_to_key_lookup(dic, val):
    for k,v in dic.items():
        if v==val:
            return k
def get_scheme(scheme):
    return val_to_key_lookup(INTERPOLATION_SCHEME, scheme)
loglog = get_scheme('log-log')
histogramic = get_scheme('histogram')

def get_all_csv(argv_1):
    assert not argv_1.endswith(".py"), "Please input the directory where the xs_*.csv and s_*.csv are read from."
    csv_names = glob.glob(os.path.join(argv_1,"*.csv"))
    xs, s = [], []
    for full_name in sorted(csv_names):
        file_name = os.path.basename(full_name)
        if file_name.startswith("xs") or file_name.startswith("sigma"):
            xs.append(full_name)
        elif file_name.startswith("source") or file_name.startswith("s_"):
            s.append(full_name)
    return xs, s

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
    elif scheme==3: # logx, liny
        m = dy/dlnx
        if xi==x2:
            return y1*dx + m*(x2*dlnx-dx)
        else:
            return y1*x_ + m*(xi*(lnxi-lnx1)- x_)
            return (y1 - m*lnx1)*x_ + m*(-x_+xi*lnxi-x1*lnx1)
    elif scheme==4: # linx, logy
        m = dlny/dx
        if xi==x2:
            return 1/m *dy
        else:
            return 1/m *y1*(exp(x_*m)-1)
    elif scheme==5:
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


def easy_logspace(start, stop, **kwargs):
    logstart, logstop = np.log10([start, stop])
    return np.logspace(logstart, logstop, **kwargs)

def plain_name(full_file_path):
    file_name = os.path.basename(full_file_path)
    short_fname = file_name.lstrip("xs_").lstrip("s_").lstrip("source_").lstrip("s_")
    return short_fname.rstrip(".csv").replace("_", ".")

def generate_group_structure(start, stop, **kwargs):
    spacing = easy_logspace(start, stop, endpoint=True, **kwargs)
    group_structure = []
    for i in range(len(spacing)-1):
        group_structure.append([spacing[i], spacing[i+1]])
    return group_structure

def percentile_of(minmax_of_range, percentile=50):
    perc = float(percentile/100)
    return min(minmax_of_range) + perc*abs(np.diff(minmax_of_range))

def convert_to_func(x,y , scheme): # same scheme over the entire thing
    return openmc.data.Tabulated1D(x, y, breakpoints=[len(x),],  interpolation=[scheme,])

def read_gs(dir_path):
    df = pd.read_csv(os.path.join(dir_path, "gs.csv"), index_col=[0])
    return df

def integrate_continuous_flux(continuous_func, gs):
    inte = Integrate(continuous_func)
    return [inte(*minmax) for minmax in gs]

def flux_conversion(flux_in, gs_in_eV, in_fmt, out_fmt):
    if isinstance(flux_in, pd.DataFrame) or isinstance(flux_in, pd.Series): # check type
        flux= flux_in.values.T
    else:
        flux=flux_in
    #convert all of them to per eV
    if in_fmt=='per MeV':
        flux_per_eV = flux/MeV
    elif in_fmt=='integrated':
        flux_per_eV = flux/np.diff(gs_in_eV, axis=1).flatten()
    elif in_fmt=='PUL':
        leth_space = np.diff(ln(gs_in_eV), axis=1).flatten()
        flux_integrated = flux*leth_space
        flux_per_eV = flux_conversion(flux_integrated, gs_in_eV, 'integrated', 'per eV')
    elif in_fmt=='per keV':
        flux_per_eV = flux/1E3
    else:
        assert in_fmt=="per eV", "the input format 'i' must be one of the following 4='integrated'|'PUL'(per unit lethargy)|'per (M)eV'"
        flux_per_eV = flux
    # convert from per eV back into 
    if out_fmt=='per MeV':
        flux_out= flux_per_eV*MeV
    elif out_fmt=='integrated':
        flux_out= flux_per_eV * np.diff(gs_in_eV, axis=1).flatten()
    elif out_fmt=='PUL':
        leth_space = np.diff(ln(gs_in_eV), axis=1).flatten()
        flux_integrated = flux_conversion(flux_per_eV, gs_in_eV,'per eV', 'integrated')
        flux_out= flux_integrated/leth_space
    else:
        assert out_fmt=='per eV', "the input format 'i' must be one of the following 4='integrated'|'PUL'(per unit lethargy)|'per (M)eV'"
        flux_out= flux_per_eV
    
    #give it back as the original type
    if isinstance(flux_in, pd.DataFrame) or isinstance(flux_in, pd.Series):
        flux_out = type(flux_in)(flux_out)
        name_or_col = "column" if isinstance(flux_in, pd.DataFrame) else "name"
        setattr(flux_out, name_or_col, getattr(flux_in, name_or_col))
    return flux_out

def get_integrated_apriori_value_only(folder):
    full_file_path = os.path.join(folder, "integrated_apriori.csv")
    integrated_apriori = pd.read_csv(full_file_path, sep="±|,", engine='python')
    return integrated_apriori['value'].values

def get_continuous_flux(folder):
    full_file_path = os.path.join(folder, "continuous_apriori.pkl")
    with open(full_file_path,'rb') as f:
        apriori_func = pickle.load(f)
    return apriori_func

def get_gs_ary(folder):
    full_file_path = os.path.join(folder, "gs.csv")
    gs = pd.read_csv(full_file_path)
    return gs.values    

print("INFO: Using point-wise input assumes the flux is continuous.")
print("Please make sure that there is a valid reason to interpolate the flux as such,")
print("e.g. the flux is from a source with little to no down-scattering.")
print("(Down-scattering destroys continuity.)")

def from_pointwise_per_eV(argv_1):
    cross_secs, sources = get_all_csv(argv_1)
    xs_df, source_df = [], []

    for source in sources:
        df = pd.read_csv(source)
        if 'MeV' in source:
            df[df.columns[0]] = df[df.columns[0]]*MeV
            df[df.columns[1]] = df[df.columns[1]]/MeV
        source_df.append(df)
    for xs in cross_secs:
        df = pd.read_csv(xs)
        # if 'MeV' in xs:
        #     df[df.columns[0]] = df[df.columns[0]]*MeV
        #     df[df.columns[1]] = df[df.columns[1]]/MeV
        xs_df.append(df)
    #do the plotting
    fig, axl = plt.subplots()
    
    axr = axl.twinx()
    for i in range(len(source_df)):
        x, y = source_df[i].values.T
        axr.plot(x, y, label=plain_name(sources[i]))
    axr.legend()
    axr.set_ylabel(rf"source flux (n/(sr eV $\mu$C))")

    for i in range(len(xs_df)):
        axl.plot(*xs_df[i].values.T, label=plain_name(cross_secs[i]), color="C"+str(i+len(source_df)))
    axl.legend(loc='upper left')
    axl.set_ylabel("cross-section (barns)")
    axl.set_xlabel("neutron energy (eV)")
    # plot a test group structure
    if len(source_df):
        source_min = min([df[df.columns[0]].min() for df in source_df])
        source_max = max([df[df.columns[0]].max() for df in source_df])

    if len(xs_df):
        xs_min = max([df[df.columns[0]].min() for df in xs_df])
        xs_max = max([df[df.columns[0]].max() for df in xs_df])

    if len(xs_df)>0 and len(source_df)>0:
        xmin, xmax = max([xs_min, source_min]), min([xs_max, source_max])
    elif len(xs_df)==0:
        xmin, xmax = source_min, source_max
    elif len(source_df)==0:
        xmin, xmax = xs_min, xs_max
    extended_range = [percentile_of([xmin, xmax], -5), percentile_of([xmin, xmax], 105)]
    axl.set_xlim(*extended_range)
    group_structure = generate_group_structure(xmin, xmax, num=RESOLUTION)
    [
    # [3.43573E6, 3.44456E6],
    # [3.44456E6, 3.64516E6],
    # [3.64516E6, 3.64731E6],
    # [3.64731E6, 3.74375E6],
    # [3.74375E6, ],
    ]

    ybounds = axl.get_ybound()
    yheight = np.diff(ybounds[::-1])/4
    for minmax in group_structure:
        # hack by using an error bar to do what it's not intended to: denote the range of the bin.
        axl.errorbar(percentile_of(minmax), percentile_of(ybounds, 10), xerr= np.diff(minmax[::-1])/2, capsize=30, color='black')

    plt.title("Group structure used for this reaction")
    plt.show()
    group_struct_df = pd.DataFrame(group_structure, columns=['min','max'])
    group_struct_df.to_csv(os.path.join(argv_1,"gs.csv"), index=False)
    
    #Now do the conversion
    pointwise_apriori_flux_per_eV = source_df[CHOICE]
    pointwise_apriori_flux_per_eV[pointwise_apriori_flux_per_eV.columns[0]] = pointwise_apriori_flux_per_eV[pointwise_apriori_flux_per_eV.columns[0]] * SCALE_FACTOR
    E = np.linspace(*extended_range, 5000)
            
    apriori_func = convert_to_func(*pointwise_apriori_flux_per_eV.values.T, scheme=get_scheme("log-log"))
    '''
    # apriori_func_linx = convert_to_func(*pointwise_apriori_flux_per_eV.values.T, scheme=get_scheme("linear-log"))
    plt.plot(E, apriori_func_linx(E))
    plt.plot(E, apriori_func_logx(E))
    plt.show()

    plt.semilogy(E, apriori_func_linx(E))
    plt.semilogy(E, apriori_func_logx(E))
    plt.show()

    plt.loglog(E, apriori_func_logx(E))
    plt.loglog(E, apriori_func_linx(E))
    plt.show()
    '''
    with open(os.path.join(argv_1, "continuous_apriori.pkl"), "wb") as f:
        pickle.dump(apriori_func, f)
    integrated_flux = integrate_continuous_flux(apriori_func, group_structure)
    if VERBOSE:
        left_edge = [gs[0] for gs in group_structure]
        plt.step(left_edge, integrated_flux, where='post')
        plt.title("flux approximation")
        plt.show()
    apriori = ary([integrated_flux, sqrt(integrated_flux)]).T
    apriori_df = pd.DataFrame(apriori, columns=['value', 'uncertainty'])
    apriori_df.to_csv(os.path.join(argv_1,'integrated_apriori.csv'), index=False)
    return apriori_df

def simple_multiply(infile, outfile, xscale_factor=1, yscale_factor=1, **kwargs):
    df = pd.read_csv(infile, **kwargs)
    col = df.columns
    df[col[0]] = df[col[0]]*xscale_factor
    df[col[1]] = df[col[1]]*yscale_factor
    df.to_csv(outfile, index=False)
    return df

if __name__=="__main__":
        if "gsMeV.csv" in [os.path.basename(i) for i in glob.glob(os.path.join(sys.argv[-1], "*"))]:
            print("gsMeV.csv detected, performing conversion")
            simple_multiply(os.path.join(sys.argv[-1],'gsMeV.csv'), os.path.join(sys.argv[-1],'gs.csv'), MeV, MeV)
        csv_files = get_all_csv(sys.argv[-1])
        if len(np.concatenate(csv_files)): # if it has any of the following files:
            '''
            "source_*.csv"
            "s_*.csv"
            "xs_*.csv"
            "sigma_*.csv"
            '''
            #then plot and get an evenly spaced group structure out of it.
            print(csv_files, "detected. Assuming they are pointwise data extracted from graph(s)."
                        +"Using log-log interpolation scheme by default."
                        +" Saving into ['gs.csv', 'integrated_apriori.csv', 'continuous_apriori.pkl']\n")
            from_pointwise_per_eV(sys.argv[-1])
        while True:
            action = input("What action would you like to perform?"
            +"(scale {file} {xscale_factor} {yscale_factor}|"
            +"convert {apriori_file} {gs_file} {in_fmt} {out_fmt}|"
            +"scaledownto {flux_file} {target_integrated_flux}|"
            +"exit)")
            sn = csv.Sniffer()

            if action=="exit":
                break
            elif "scale"==action.split()[0]:
                fname = os.path.join(sys.argv[-1], action.split()[1])
                if not sn.has_header(fname):
                    simple_multiply( fname, "new_"+fname, xscale_factor=float(action.split()[2]), yscale_factor=float(action.split()[3]), header=None)
                else:
                    simple_multiply( fname, "new_"+fname, xscale_factor=float(action.split()[2]), yscale_factor=float(action.split()[3]))
            elif "convert"==action.split()[0]:
                ap_file = os.path.join(sys.argv[-1], action.split()[1])
                gs_file = os.path.join(sys.argv[-1], action.split()[2])
                has_header = sn.has_header(ap_file)
                pd.read_csv(flux_in)
                # gs_array = os.path.join(sys.argv[-1], action.split())
            elif "scaledownto"==action.split()[0]:
                fluxfile = os.path.join(sys.argv[-1], action.split()[1])
                outfile = os.path.join(sys.argv[-1], "new_"+action.split()[1])
                flux_df = pd.read_csv(fluxfile)
                target_integrated_flux = float(action.split()[2])

                centroids = flux_df[flux_df.columns[0]].values
                gs = ary([np.hstack([0,centroids[:-1]]),centroids[:]])

                continuous_flux = convert_to_func(*flux_df.values.T, scheme=get_scheme("log-linear"))
                print(integrate_continuous_flux(continuous_flux, gs.T))
                scale_factor = target_integrated_flux/sum(integrate_continuous_flux(continuous_flux, gs.T))

                flux_df[flux_df.columns[1]] = flux_df[flux_df.columns[1]]*scale_factor
                flux_df.to_csv(outfile, index=False)

        '''
        Conclusion after examining the plots for 'Ohio/apriori':
        1. The slope of the source at any one point either stays zero across all energies of deterium, or varies across energies of deterium.
        This means that there is no benefit to parametrising (interpolating) the apriori neutron spectrum in terms of anything but histogramic distributions.
        2. The range chosen is between 
        '''