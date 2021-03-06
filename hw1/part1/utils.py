import numpy as np
import matplotlib.pyplot as plt

#############################################################################
#  Normalize features of data matrix X so that every column has zero        #
#  mean and unit variance                                                   #
#     Input:                                                                #
#     X: N x D where N is the number of rows and D is the number of         #
#        features                                                           #
#     Output: mu: D x 1 (mean of X)                                         #
#          sigma: D x 1 (std dev of X)                                      #
#         X_norm: N x D (normalized X)                                      #
#############################################################################

def feature_normalize(X):

    ########################################################################
    # TODO: modify the three lines below to return the correct values
    ########################################################################
    mu, sigma = np.mean(X, axis = 0), np.std(X, axis = 0)
    sigmaInv = np.zeros(sigma.shape)
    sigmaInv[np.nonzero(sigma)] = 1/sigma[np.nonzero(sigma)]
    X_norm = (X -mu)*sigmaInv  
    ########################################################################
    return X_norm, mu, sigma


