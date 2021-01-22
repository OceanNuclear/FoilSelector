from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np
from numpy import log as ln
import numpy.linalg as la
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import os, sys, itertools, functools
from math import factorial as fac
from collections import OrderedDict
from tqdm import tqdm
from scipy.constants import N_A # Avogadro's number
from scipy.special import comb as n_choose_k
from scipy.special import perm as n_perm_k
from misc_library import get_apriori, unserialize_dict, save_parameters_as_json, ordered_set
from uncertainties.unumpy import nominal_values
from GetReactionRates import IRRADIATION_DURATION, MEASUREMENT_DURATION, FOIL_AREA, BARN, MM_CM

SATURATION_COUNT_RATE = 1000 # maximum number of gamma countable accurately per second
MAX_THICKNESS = 0.1 # mm
"""
if RANK_BY_DETERMINANT: calculate a non-singular matrix representing the curvature,
And then performance of each set of foil combination is then quantified by the determinant of its curvature matrix.
else: use various approach.
"""
COUNT_THRESHOLD = 300 # reactions with less than this many count will not be counted.
RANK_BY_DETERMINANT = False

def D_KL(test_spectrum, apriori_spectrum):
    """ Calculates the Kullback-Leibler divergence.
    """
    fDEF = apriori_spectrum/sum(apriori_spectrum)
    f = test_spectrum/sum(test_spectrum)
    from autograd import numpy as agnp
    ratio = agnp.nan_to_num(fDEF/f)
    return agnp.dot(fDEF, ratio)
    
def curvature_matrix(R, S_N_inv):
    """
    parameters
    ----------
    R : response matrix, with the appropriate thickness information alreay included.
    S_N_inv : inverse of the covariance matrix of the reaction rates vector N
    """
    return R.T @ S_N_inv @ R

def confusion_matrix(R):
    """
    input: Response matrix (m*n), no need to normalize. THe thickness does not matter.
    output: confusion matrix, dimension = (n*n) where n = number of bins. 
    """
    return la.pinv(R) @ R

def confusion_matrix_diag_sum(R):
    return np.diag(la.pinv(R) @ R).sum()/R.shape[1]

def scalar_curvature(R, apriori, threshold_count=COUNT_THRESHOLD):
    """
    Assume no correlations between different reaction channels, which, of course is a big lie, but I'll think about the consequence of this lie later.
    This function takes the del^2 chi2.
    This is different from taking the determinant of the chi2 curvature matrix.
    """
    counts = R @ apriori
    detectable_reactions = counts>threshold_count
    R_mod = R[detectable_reactions]
    S_N = np.diag(counts[detectable_reactions])
    return np.diag(R_mod.T @ la.inv(S_N) @ R_mod).sum()

def show_detectable_reactions(R, apriori, threshold_count=COUNT_THRESHOLD):
    counts = R @ apriori
    detectable_reactions = counts>threshold_count
    return detectable_reactions

class Foil(object):
    """
    A collection of all of the reactions in a foil.
    """
    __slots__ = ["partial_density","thickness","material_name","response_per_mm","counts_per_primary_product","gamma_counts_response_per_mm","counts_response", "reaction_names"]
    def __init__(self, material_name, density, microscopic_cross_sections, counts_per_primary_product, reaction_names=[]):
        """
        Determines the optimal thickness for said foil as well.
        """
        self.material_name = material_name
        self.partial_density = density
        self.response_per_mm = (microscopic_cross_sections.values.T / MM_CM * BARN * self.partial_density.values).T # reactions per cm^-2 fluence per mm thickness
        self.counts_per_primary_product = counts_per_primary_product
        self.gamma_counts_response_per_mm = (self.response_per_mm.T * self.counts_per_primary_product.values).T * FOIL_AREA # gamma counts fluence per mm thickness
        total_counts_per_mm_thickness = self.gamma_counts_response_per_mm @ apriori_flux
        # choose a thickness: maximize foil thickness without saturating the detector
        self.thickness = np.clip(
            SATURATION_COUNT_RATE*MEASUREMENT_DURATION/total_counts_per_mm_thickness.sum(), 0, MAX_THICKNESS
            )
        self.counts_response = self.gamma_counts_response_per_mm * self.thickness # gamma counts per foil
        self.reaction_names = reaction_names

    def __repr__(self):
        return super().__repr__().replace("Foil object", self.material_name+" foil")

    def __add__(self, foil_like):
        assert isinstance(foil_like, (Foil, FoilSet)), "Can only add Foil/FoilSet onto another Foil/FoilSet to create another FoilSet."
        return FoilSet(self, foil_like)

    def __copy__(self):
        new_foil = self.__new__(self.__class__) # create a new instance of the class
        for attr in self.__slots__:
            setattr(new_foil, attr, getattr(self, attr))
        return new_foil

    def filter(self, apriori, threshold_count=COUNT_THRESHOLD):
        filtered_foil = self.__copy__()
        # remove the unwanted reactions (too low number of counts)
        filtered_reactions = show_detectable_reactions(filtered_foil.counts_response, apriori, threshold_count)
        filtered_foil.response_per_mm               = filtered_foil.response_per_mm[filtered_reactions]
        filtered_foil.counts_per_primary_product    = filtered_foil.counts_per_primary_product[filtered_reactions]
        filtered_foil.gamma_counts_response_per_mm  = filtered_foil.gamma_counts_response_per_mm[filtered_reactions]
        filtered_foil.counts_response               = filtered_foil.counts_response[filtered_reactions]
        filtered_foil.reaction_names                = ary(filtered_foil.reaction_names)[filtered_reactions].tolist()
        return filtered_foil

class FoilSet():
    """a collection of Foils.
    The reason why Foil *isn't* the parent of FoilSet is because Foil contains attributes that FoilSet doesn't need: 
    response_per_mm, counts_per_primary_product, gamma_counts_response_per_mm, are all values that are meaningless once thickness is chosen.
    """
    def __init__(self, *foils):
        self.counts_response, self.material_name, self.thickness = [], [], []
        for foil_like in foils:
            self.counts_response.append(foil_like.counts_response)
            if isinstance(foil_like,Foil):
                self.material_name.append(foil_like.material_name)
                self.thickness.append(foil_like.thickness)
            elif isinstance(foil_like,FoilSet):
                self.material_name += foil_like.material_name
                self.thickness += foil_like.thickness
            else:
                raise TypeError(f"Only as foil_like objects ({Foil}, {FoilSet}) are accepted; {type(foil_like)} is not allowed!")
        self.counts_response = np.concatenate(self.counts_response)

    def __repr__(self):
        return super().__repr__().replace("FoilSet object", "FoilSet of {} foils".format(len(self.material_name)))

    def __add__(self, foil_like):
        assert isinstance(foil_like, (Foil, FoilSet)), "Can only add Foil/FoilSet onto another Foil/FoilSet to create another FoilSet."
        return FoilSet(self, foil_like)

class CombinationNode(object):
    def __init__(self, occupancy, plot_coordinates=()):
        self.occupancy = occupancy
        self.parent = []
        self.plot_coordinates = plot_coordinates

    def movaibility_to_the_right():
        """
        Return array saying which bit can move right
        """
        return np.hstack([np.diff(self.occupancy)==-1, False])

    @property
    def occupancy(self):
        return self._occupancy

    @occupancy.setter
    def occupancy(self, _occupancy):
        self._occupancy = ary(_occupancy, dtype=bool)
        numbering = np.cumsum(self._occupancy)
        numbering[self._occupancy] = int(0)
        self.numbered_occupancy = numbering

def get_foilset_condensed_name(foil_list, space_then_symbol_in_bracket=True):
    if space_then_symbol_in_bracket:
        return "-".join(i.split()[1].strip("()") for i in foil_list)
    else:
        return "-".join(i[:2] for i in foil_list)

if RANK_BY_DETERMINANT:
    import autograd
    D_KL_hessian_getter = autograd.hessian(lambda x: D_KL(x, apriori_fluence.copy()))
    
def det_curvature(R, apriori, include_D_KL_contribution=RANK_BY_DETERMINANT, custom_hessian_contribution=None):
    hess_matrix = curvature_matrix(R, la.inv(np.diag(R @ apriori_fluence)))
    if include_D_KL_contribution:
        hess_matrix += D_KL_hessian_getter(apriori_fluence)
    if custom_hessian_contribution:
        hess_matrix += custom_hessian_contribution
    return la.det(hess_matrix)

__add__ = lambda x, y: x+y
chain_sum = lambda iterable: functools.reduce(__add__, iterable)

descending_odict = lambda d: OrderedDict(sorted(d.items(), key=lambda key_val: key_val[1], reverse=True))

def choose_top_n_pretty(func, target_chosen_length, policy, choices, verbose=True):
    # number of possible reactions = 
    num_combinations = int(n_choose_k(len(choices), target_chosen_length))
    if type(policy)==int:
        print("Attempting to choose an optimal combination of foils by choosing the top {} foils at every move, in the solution space of {} possible combinations".format(policy, num_combinations))
        print("which should return a dictionary of length with an upper limit = {}".format(policy, np.clip(policy**target_chosen_length, num_combinations, None)))
        print("This can take up to {} evaluations ...".format( min([policy**target_chosen_length, int(n_perm_k(len(choices), target_chosen_length))]) ))
    elif type(policy)==float:
        print("Attempting to choose an optimal combination of foils by choosing the top {} % of foils, in the solution space of {} possible combinations.".format(policy, num_combinations))
    return choose_top_n(func, target_chosen_length, policy, choices, verbose=verbose)

def choose_top_n(func, target_chosen_length, policy, choices, chosen=set(), verbose=True):
    """
    Find a chosen combination of strings that give the highest output value when plugged into func,
    using the "greedy" step approach, taking the step that leads to the greatest increase in the output.
    ("Taking a step" here refers to updating the set of strings by adding one element from "choices" to "chosen".)

    Parameters
    ----------
    func: function that takes in a list of strings (all of which are present in the list variable choices) in any arbitrary order and output a scalar.
    target_chosen_length: number of elements to be placed.
    policy:
        if int: n = int(policy), n must be >=1. Will take a step forward in each of the top n most promising steps.
        if float: f = float(policy), f must be between 0 and 1. Will take a step forward on each of the top 100*f% most promising steps.

    returns
    """
    # calculate scalar outputs for all possible steps
    output_scalars = {name:func(name, *chosen) for name in choices}
    # and then choose the names to be used as the next steps.
    if type(policy)==int:
        output_scalars = descending_odict(output_scalars)
        next_steps = OrderedDict((name, out) for name, out in list(output_scalars.items())[:policy]) # names of the chosen elements
    elif type(policy)==float and (0<=policy<=1):
        out_max, out_min = max(output_scalars.values()), min(output_scalars.values())
        diff = out_max - out_min
        threshold = out_max - policy*diff
        next_steps = OrderedDict((name, out) for name, out in output_scalars.items() if out>=threshold)
    else:
        raise ValueError("policy must be either: 1. choose the top n most promising steps (int), or 2. choose the top (100*f)% most promising steps (0<=float<=1)")

    # termination condition
    if (len(chosen)+1)>=target_chosen_length:
        if verbose:
            for name, out in next_steps.items():
                print("Branch ends! Foils chosen in this branch = {} with value = {}".format([name, *chosen], out))
        return { tuple(sorted([name, *chosen])): out for name, out in next_steps.items() }

    # recursion
    else:
        combos_used_and_their_output = OrderedDict()
        for name in next_steps.keys():
            added_chosen, remaining_choices = chosen.copy(), choices.copy()
            # transfer the chosen name into the added set; and remove it from the remaining_choices set.
            added_chosen.add(name), remaining_choices.remove(name)
            if verbose:
                print("foils chosen in this branch so far = {} with value = {}".format(added_chosen, next_steps[name]))
            combos_used_and_their_output.update( choose_top_n(func, target_chosen_length, policy, remaining_choices, added_chosen, verbose) )
        return combos_used_and_their_output

def choose_top_1(func, target_chosen_length, choices, chosen=[], verbose=True):
    """
    Same as choose_top_n, but policy is fixed at int(1) and the chosen is now a list, allowing the order in which items were added to be preserved.
    """
    print(f"{func=}")
    print(f"{target_chosen_length=}")
    print(f"{choices=}")
    print(f"{chosen=}")
    print(f"{verbose=}")
    output_scalars = {name:func(name, *chosen) for name in choices}
    max_val = max(output_scalars.values())
    # reverse lookup dict to find what name gave the max output value.
    name = [name for name, out in output_scalars.items() if out==max_val][0]
    #shift the name from choices to chosen
    chosen, choices = chosen.copy(), choices.copy() # unlink from the list from the function call above.
    # If I don't unlink, it'll spazz out and reuse the old copy of chosen, choices.
    chosen.append(name)
    print(len(chosen))
    if verbose:
        print("foil chosen so far =", chosen)
    choices.remove(name)
    if len(chosen)>=target_chosen_length: # termination condition
        if len(chosen)>target_chosen_length:
            print("Hold up, how did you get here?")
        return (chosen, output_scalars[name])
    # recursion condition
    else:
        return choose_top_1(func, target_chosen_length, choices, chosen, verbose)

def perturb_top_1_order(func, target_chosen_length, choices, verbose=True):
    greedy_choices = choose_top_1(func, target_chosen_length, choices, verbose)[0]
    negated_choices = [c for c in choices if c not in greedy_choices]
    raise NotImplementedError("Haven't got time /idea of how to finish mixing up the two list greedy_choices and negated_choices.")

"""
There are nCk solutions.

Verify that "greedy" appraoch is a reasonable?
1. greedy appraoch -> greedy choice reordering
2. relaxation of choices: branch at each foil-num increment: 
    include n-1 more optimal choices
    -> generates at most n C k combos
3. relaxation of order: using the greedy choice ordering,
    monte carlo approach of mixing it up?
        (Because there are n! ways of mixing it up)
        genetic algorithm
"""

if __name__=='__main__':
    # get the relevant data from files.
    assert os.path.exists(os.path.join(sys.argv[-1], 'gs.csv')), "Output directory must already have gs.csv"
    gs = pd.read_csv(os.path.join(sys.argv[-1], 'gs.csv')).values

    apriori_flux, _apriori_fluence = get_apriori(sys.argv[-1], IRRADIATION_DURATION)
    apriori_fluence = np.clip(_apriori_fluence, min([i for i in _apriori_fluence if i>0])*0.5, None)

    assert os.path.exists(os.path.join(sys.argv[-1], "counts.csv")), "A 'counts.csv' file must exist (generated by GetReactionRates.py) at the target directory listed by the last argv argument."
    assert os.path.exists(os.path.join(sys.argv[-1], "response.csv")), "A 'response.csv' file must exist (generated by ReadData.py) at the target directory listed by the last argv argument."
    print("\nReading the counts and thickness data from 'counts.csv'...")
    population = pd.read_csv(os.path.join(sys.argv[-1], "counts.csv"), index_col=[0])
    print("\nReading the microscopic cross-section...")
    raw_response_matrix = pd.read_csv(os.path.join(sys.argv[-1], "response.csv"), index_col=[0])
    raw_response_matrix = raw_response_matrix.loc[population.index]

    # quick rewrite of the function names and nomenclature
    get_sensitivity = lambda r: scalar_curvature(r, apriori_fluence)
    get_specificity = lambda r: confusion_matrix_diag_sum(r)

    print("\nStripping the uncertainties from the 'total gamma counts per primary product' column...")
    total_gamma_counts_per_primary_product = pd.Series(
        nominal_values(unserialize_dict(population["total gamma counts per primary product"].values.tolist())),
        index=population["total gamma counts per primary product"].index
        )

    # create the list of candidate materials
    foil_candidates = []
    material_series = population["default material"]
    print(f"\nCreating the foils, using COUNT_THRESHOLD = {COUNT_THRESHOLD}")
    material_progress_bar = tqdm(ordered_set(material_series))
    for material in material_progress_bar:
        material_progress_bar.set_description(material)
        if material=="Missing (N/A)":
            continue # skip the material "missing", which is my text equivalent of nan
        matching_material = material_series == material
        # mt_list = [parent_product_mt.split("=")[1] for parent_product_mt in population[matching_material].index]
        # mt_list = [ tuple(int(mt) for mt in mt_series.strip("()").split(",")) for mt_series in mt_list]

        foil_candidates.append(Foil(material, 
                    population["partial number density (cm^-3)"][matching_material],
                    raw_response_matrix[matching_material],
                    total_gamma_counts_per_primary_product[matching_material],
                    reaction_names=["-".join(parent_product_mt.split("-")[1:]) for parent_product_mt in population[matching_material].index],
                    )
                )
    individual_curvature = [get_sensitivity(f.counts_response) for f in foil_candidates]

    # reorder them
    foil_candidates = [ foil_candidates[ind] for ind in np.argsort(individual_curvature)[::-1] ]
    foil_candidates_dict = OrderedDict((f.material_name, f.filter(apriori_fluence)) for f in foil_candidates)
    del foil_candidates
    individual_curvature = OrderedDict((f_name, get_sensitivity(foil.counts_response)) for f_name, foil in foil_candidates_dict.items())
    individual_orthogonality = OrderedDict((f_name, get_specificity(foil.counts_response)) for f_name, foil in foil_candidates_dict.items())
    num_reactions = OrderedDict((f_name, foil.counts_response.shape[0]) for f_name, foil in foil_candidates_dict.items())
    complete_foil_set = FoilSet(* list(foil_candidates_dict.values()) )
    thickness = OrderedDict((f_name, foil.thickness) for f_name, foil in foil_candidates_dict.items())
    foil_data_df = pd.DataFrame([thickness, num_reactions, individual_curvature, individual_orthogonality],
        index=["thickness", "number of reactions", "individual sensitivity", "individual specificity"]).T
    foil_data_df["number of reactions"] = foil_data_df["number of reactions"].astype("int")
    with pd.option_context("display.max_rows", None):
        print(foil_data_df)

    # perform optimization
    @functools.lru_cache(maxsize=2**31)
    def get_specificity_from_names(*names):
        foil_set = chain_sum([foil_candidates_dict[name] for name in names])
        return get_specificity(foil_set.counts_response)

    def get_specificity_from_unsorted_names(*names):
        return get_specificity_from_names(*sorted(names))

    # calculate the specificity and senitivity for each foilset, and then rank them.
    if SCATTERPLOT:=False:
        specificity, sensitivity = {}, {}
        # generate a complete set of solution
        num_foils_chosen = 3 # honestly, 71 choose 3 is a huge number so I really dont' want to make it any bigger.
        # It typically runs through 67 combinations per second.
        num_combinations = int(fac(len(foil_candidates_dict)) / (fac(len(foil_candidates_dict)-num_foils_chosen) * fac(num_foils_chosen)))
        print("Creating the scatter plot of sensitivity vs specificity, one data point for each of the {} choose {} = {} combination of foils:".format(len(foil_candidates_dict),
                                                                                                                                                        num_foils_chosen, num_combinations))
        for f_set in tqdm(itertools.combinations(foil_candidates_dict.values(), num_foils_chosen), total=num_combinations):
            f_set = chain_sum(f_set)
            f_set_name = get_foilset_condensed_name(f_set.material_name)
            specificity[f_set_name] = get_specificity(f_set.counts_response)
            sensitivity[f_set_name] = get_sensitivity(f_set.counts_response)
        
        metrics = pd.DataFrame([sensitivity, specificity], ["sensitivity", "specificity"]).T
        plt.scatter(*metrics.values.T, marker='x', alpha=0.8)
        plt.xlim( 0.8*metrics["sensitivity"].min(), 1.2*metrics["sensitivity"].max() )
        plt.xlabel("sensitivity (sum of diagonal of the curvature matrix)")
        plt.ylabel("specificity")
        plt.title("{} foils choose {} = {} combinations possible ".format(len(foil_candidates_dict), num_foils_chosen, num_combinations))
        plt.show()

    print(choose_top_1(get_specificity_from_unsorted_names, 10, list(foil_candidates_dict.keys())))
    print(choose_top_1(get_specificity_from_unsorted_names, 10, list(foil_candidates_dict.keys())))
    print(choose_top_1(get_specificity_from_unsorted_names, 10, list(foil_candidates_dict.keys())))
    print(choose_top_1(get_specificity_from_unsorted_names, 10, list(foil_candidates_dict.keys())))
    print(choose_top_1(get_specificity_from_unsorted_names, 10, list(foil_candidates_dict.keys())))
    print(get_specificity_from_names.cache_info())
    get_specificity_from_names.cache_clear()
    # print(choose_top_n_pretty(get_specificity_from_unsorted_names, 10, 1, list(foil_candidates_dict.keys())))
    # print(get_specificity_from_names.cache_info())
    # print(choose_top_n_pretty(get_specificity_from_unsorted_names, 10, 1, list(foil_candidates_dict.keys())))
    # print(get_specificity_from_names.cache_info())
    # print(choose_top_n_pretty(get_specificity_from_unsorted_names, 10, 1, list(foil_candidates_dict.keys())))
    # print(get_specificity_from_names.cache_info())
    print(choose_top_1(get_specificity_from_unsorted_names, 10, list(foil_candidates_dict.keys())))
    print(get_specificity_from_names.cache_info())
    print(choose_top_n)
    sys.exit()
    save_parameters_as_json(sys.argv[-1], dict(SATURATION_COUNT_RATE=SATURATION_COUNT_RATE, MAX_THICKNESS=MAX_THICKNESS, COUNT_THRESHOLD=COUNT_THRESHOLD, RANK_BY_DETERMINANT=RANK_BY_DETERMINANT))

    # save parameters at the end.

    try:
        f_set = complete_foil_set
        while True:
            R = f_set.counts_response
            S_N_inv = la.inv(np.diag(R @ apriori_fluence))

            # confusion matrix
            sns.heatmap(confusion_matrix(R))
            plt.title(f"Confusion matrix of the foilset with {len(f_set.material_name)} foils")
            plt.show()

            # curvature, presentation 1 (bar plot)
            full_curvature = curvature_matrix(R, S_N_inv)
            curvature_diag = np.diag(full_curvature)
            plt.bar(range(len(curvature_diag)), curvature_diag)
            plt.suptitle("Curvature of chi2 landscape in the principal directions,")
            plt.title("plotted by bin number")
            plt.yscale('log')
            plt.xlabel("bin number")
            plt.show()

            # curvature, presentation 2 (loglog line plot)
            plt.loglog(gs.flatten(), np.repeat(curvature_diag, 2))
            plt.title("Curvature of chi2 landscape in the principal directions")
            plt.xlabel("E (eV)")
            plt.ylabel("Curvature ((std score)^2/unit fluence(cm^-2))")
            plt.show()

            # curvature, presentation 3 (heatmap)
            sns.heatmap(full_curvature)
            plt.title("Curvature of chi2 landscape")
            plt.show()

            # curvature, presentation 2.5 (incremental)
            for foil_name in f_set.material_name:
                r_line = foil_candidates_dict[foil_name].counts_response
                S_N_inv = la.inv(np.diag(r_line @ apriori_fluence))
                curvature_diag = np.diag(curvature_matrix(r_line, S_N_inv))
                plt.loglog( gs.flatten(), np.repeat(curvature_diag, 2), label=foil_name)
            plt.loglog(gs.flatten(), np.repeat(curvature_diag, 2), label="Total")
            plt.suptitle("Curvature of the chi2 landscape in the principal directions")
            plt.title("including partial contributions from each")
            plt.legend()
            plt.show()

            # Robin's suggestion
            plt.loglog( (R.T * apriori_fluence).T )
            plt.title("Partial reactionr ates")
            plt.show()

            print("Attempting to pick a second selection ...")
            print("Ignore which of the following foils?")
            print([f_name for f_name in foil_candidates_dict.keys()])
            ignored_foils = input()
            f_set = FoilSet(*[foil for f_name, foil in foil_candidates_dict.items() if f_name not in ignored_foils])

            print("Foils used :\n", "\n".join(f_set.material_name))
    except KeyboardInterrupt:
        print("Terminating...")
    if RANK_BY_DETERMINANT:
        raise NotImplementedError("This feature hasn't been implemented yet!")

