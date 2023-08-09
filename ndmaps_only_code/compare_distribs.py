import numpy as np

from scipy.stats import shapiro
from numpy.random import normal


def KStest_raderrs(vals,err,mean):
    '''Check of whether the data in radial bins are normally distributed
    Input is the specific values of the profile at a given radius (not specified)
    and the assumed Gaussian mean and error of the average.
    Output is the value of the KS statistic and the p-value'''
    from scipy.stats import ks_2samp

    data = normal(loc=mean, scale=err, size=len(vals))
    stat, pval = ks_2samp(vals, data)

    if pval < 0.05:
        same=False
        # print(f'p-val is {pval}, so not the same distribution buddy')
    else:
        same=True
        #print(f'p-val is {pval}, so these are from the same distribution')

    return stat, pval, same