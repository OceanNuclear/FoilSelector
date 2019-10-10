# FoilSelector
Activation foil Selector for fusion reactor's neutron spectrum unfolding.

Foil Selector tool for choosing the type of foil to be placed in the reactor in JET, ITER, DEMO, and other future nuclear fusion power plants for the purpose of measuring neutron spectra via unfolding.

## Data preparation
- > User action required: Unzip .zip file containing nuclear data in the ```RawCrossSection/``` directory, as endf format.
1.	Read the list of elements, their elemental fractions, and their respective reactions.
2.	Find the corresponding cross-sections in the data file stored in the same directory, for each of the reactions.
3.	Log-log Interpolate into the Vitamin-J group structure. (Use the default unit, i.e. barns)
4.	Save the list of reactions as dataframe, indexed by the reaction name. i.e. ```X-99(n,?)Y``` without header.

## Selection by physical property
- > User action required: copy the list of reactions generated in the folder of the [last step](#Data-preparation).
- > User action required: provide the resolution curve as a 2-column file.
- > User action required: provide the hazard file.

1.	Remove the fissile/fertile parents, fissile daughters/fast decaying parents from the list of nuclide read.
2.	Automatically rule out those which are fissile/fertile/flamable/explosive/highly reactive.
3.	Print them.
	- > User action requried: replace them with chemically stable and radiologically safe alternatives.
4.	List the daughter's names, half-lives, hazards.
5.	Give warning if the half-lives are too short, i.e. can only be used in the rabbit system.
	- > (optional) User action requried: set upper and lower threshold for half-lives.
	- > User action requried: ignore reactions which are too short lived.
6.	Read the efficiency curve and calculate the peak span of each (lower-upper limit)
7.	Give warning if there are overlapping peaks from the same element. (print half-live of all conflicting peaks)
8.	Plot the response function (with unity area, for ease of examination) in both linear y-scale and log y-scale.
9.	Save the (ppm), (branching fraction per nuclide), (ppm * branching fraction) of each nuclide in the final material, as a csv file; because these are the only constants that will be re-used regardless of the activation system design.

## Selection by effectiveness
Get the number of pulse detected by the gamma detector at the photo peak per neutron irradiated towards the foil
- > User action required: Give the neutron spectrum (neutron fluence of each bin) and intrinsic efficiency, each as a 2-column files.
- > (Developer action required: Save a reference of each of these things ^ at the top directory)
- > (Developer action required: May want to expand on this functionality in the future to increase flexibility of input files, i.e. include error)
1.	Read the (intrinsic photopeak efficiency) at the given gamma peak energy

For each reaction,
2.	Convolute flux (assume a single, rapid pulse of irradiation) onto microscopic cross-sections to get activity induced per parent nuclide, Z

3.	Multiply by Number = (V * N<sub>d</sub> ) to get total initial activity A<sub>∞</sub>
4.	Multiply by (decay correction factor) to get total activity at count time A<sub>0</sub>
	- Use fispact modelling to get decay correction factor?
5.	Multiply by (ppm * branching fraction) * (max. abs. eff. of setup) = (ppm * branching fraction) * (intrinsic photopeak efficiency * geometric correction factor(roughly =1/2) ) => Count Rate C<sub>0</sub>
6.	Sum across the entire element to ensure dead time is within acceptable limit.
	- Else change the (geometric correction factor) OR (volume) by going back to #3.
7.	Multiply onto (measurement period) (or integrate to account for decay during measurement) => Total Counts
	-	> User action required: irradiation period and count time of **each element** should be specifed by user.
 		- if decay is known to be fast, has to choose a short count time (10 mins). Otherwise standardize to the same irradiation period.
8.	Calculate the uncertainty by sqrt(total count) 
- (May minimize the uncertainty by some method in the future)

9.	Propagate uncertainty backwards by 
	- (measurement period)
	- (decay correction factor)
	- ()
10.	List the constant of proportionality between each factor and the total response rate (by differentiation) used above and the flux used, and therefore how to reduce the uncertainty.
	- (remember to convert back to non-PUL before summation to get the total response rate)
- (Conserved quantity at this stage: Number * geometric correction factor * something else = the total number of gammas detected by the detector at photopeak per neutron irradiated)

## Increase counting efficiency
1.	Check for overlapping peaks again, but this time it's within the entire list of neutron detection foils.
2.	List the combination of foils that have overlapping peaks at all.
3.	State to the user that "all other foils can have their peaks acqured together (as long as the dead-time is not too high)".

## End of run
1.	Show user the result, tell them the effectiveness of each foil.
2.	Show the list of peaks associated with each parent and daughter, and their half-lives.
3.	Save the microscopic cross-section matrix.
4.	Save the 
- > User action required: re-run the program from [step 1](#Data-preparation) after removing the least effective ones.

## Challenges
1.	~~Nuclides must maintain their un-enriched, elemental ratio~~ ([dealt with in step 2](#Selection-by-physical-property))
2.	~~For each type of material, required count rate must be settable.~~ ([dealt with in step 3](#Selection-by-effectiveness))
	- So to achieve min. uncertainty without exceeding the deadtime of the detector.
3.	~~Should also compile the list of alpha- and neutron-emitting reactions; discard if neutron production of large enough dose is found; give user a warning if alpha is found.~~ ([dealt with in step 2](#Selection-by-physical-property))
4.	~~Must have medium half-lives~~ ([dealt with in step 2](#Selection-by-physical-property))
5.	~~Gamma lines do not overlap~~ ([dealt with in step 2](#Selection-by-physical-property))
6.	If peaks overlap within the same parent element, need to use half-lives to differentiate them; or ignore if one has very small Z/atom=∫σφ compare to the other one's.
7.	May consider using a thermal shield? i.e. Gd, Cd, B.
8.	How to shield against proton/deuterium activation?
	- If not possible, how to account for/minimize them?
9.	Need to account for breeding of parent nuclides after neutron irradiation.
10.	Need to prove/justify that geometric mean is the best way of interpolating the microscopic cross-sections into its own discretized version (particularly at the the resonance peaks).

## Design choices
1.	Stick to only the Vitamin-J group structure right now; increase flexibility later.
2.	Choices of conserved quantity:
	- ~~Area under response function,~~ (irrelevant in the later stage)
	- std(Z<sub>i</sub>) (when the response functions are scaled such that uncertainty of Z<sub>i</sub> is constant, we can directly examine the effectiveness of each respones function by its overall magnitude, and the singular values of the matrix generated.)
	- ~~std(Z<sub>i</sub>)/Z<sub>i</sub>~~ (irrelevant)
3.	Work in PUL unit.
4.	Assume that the shape of the neutron flux distribution in energy space does not vary; it simply goes up and down with the total flux. (Alternative: can fit empirical function describing relationship between total neutron flux and skewedness of the flux.)
5.	Use all the peaks avaiable for every daughter nuclide created.
6.	Maximize the volume to reduce the uncertainty in number of counts.