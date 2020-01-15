from numpy import cos, arccos, sin, arctan, tan, pi, sqrt; from numpy import array as ary; import numpy as np
from numpy.linalg import inv, pinv, inv
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

import os, sys
try:
    os.chdir(sys.argv[1])
except IndexError:
    print(f"usage: python3 {sys.argv[0]} directory_containing_individual_xs_csv/")

# response_matrix=[]
# for cross_sec_file in [i for i in os.listdir('.') if i.endswith(".csv") ]:
#     response_matrix.append( pd.read_csv(cross_sec_file, header=None).values.reshape([-1]) )
#     reaction_name = cross_sec_file.split('.')[0]
#     print(f'The cross section of {reaction_name} has been read.')
    
def SimpleCurvature(R, sigma):
    return (R.T**2/sigma**2).T
    
R=  pd.read_csv('R_adjusted.csv', header=None, index_col=0).values
sigma = np.ones(11)*0.05

def CovarianceMatrix_inv(R, sigma):
    S_z = np.diag(sigma**2)
    S_z_inv = inv(S_z)
    already_inverted_S_phi= R.T.dot(S_z_inv).dot(R)
    return already_inverted_S_phi

def ConfusionMatrix(R, sigma, phi_0, option='lin-lin'):
    phi_theo = phi_0 # by choice, we assume that the apriori is spot on
    Z_theo = R.dot(phi_theo)
    Z_meas = Z_theo.copy() #again, by choice, we assume that the measured reaction rate does not deviate from the theoretical reaction rates.
    phi_g_minus_1 = phi_0
    Z_g_minus_1 = Z_meas
    RHSvec = ( Z_g_minus_1/sigma**2 ).dot(R)
    S_inv = CovarianceMatrix_inv(R, sigma)
    
    if option=='lin-lin':
        Delta_g_minus_1 = pinv(S_inv).dot(S_inv)
    return Delta_g_minus_1

apriori=np.ones(len(R.T))
print(ConfusionMatrix(R, sigma, apriori)) #sigma here is error on Z_0, i.e. the error on the measured reaction rates if the measured reaction rates is identical to the theoretical reaction rates created by the ttrue spectrum