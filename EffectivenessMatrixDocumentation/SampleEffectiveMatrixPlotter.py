from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np; tau = 2*pi
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import norm
WIDTH = 10
NUM_BINS = 175
NUM_REACT= 11

step_size = NUM_BINS/NUM_REACT
R = np.zeros([NUM_REACT, NUM_BINS])
centroid = 0 #set the first reaction's centroid to equal zero.
for row in range(NUM_REACT):
    i = np.arange(NUM_BINS)
    dist = norm.pdf(i, centroid, WIDTH)
    R[row] = dist/max(dist)
    centroid += step_size
np.random.seed(0)
np.random.shuffle(R)

ax = sns.heatmap(R, cmap='coolwarm')
ax.figure.set_size_inches(14,6)
plt.title("Effective response matrix")
plt.ylabel("Reaction number")
plt.xlabel("energy bin number")
plt.savefig("EffectiveResponseMatrix.png", )