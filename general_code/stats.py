import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import scipy.stats as stats
from numpy.random import normal
from functools import partial

def ratio_covmat(vec1, vec2, samples1, samples2, weights=None):
    # calculates the covariance matrix for a ratio of two vectors
    # samples1 and samples2 are arrays with each row corresponding to a variable,
    # each column corresponding to an observation
    rat = vec1/vec2
    if weights is None:
        weights = np.ones(samples1.shape[1])
    
    datavec_samples = np.vstack((samples1, samples2)) # should be an array 2*len(vec1) x len(samples1)
    
    var12_covmat = np.cov(datavec_samples, aweights=weights) # covmat of var1(r1), var1(r2), etc for variable 1
    
    rat_covmat  = np.zeros((len(vec1), len(vec1)))
    for i in range(len(vec1)):
        for j in range(len(vec1)):
            rat_covmat[i,j] = 1/(vec2[i]*vec2[j])*(var12_covmat[i,j]-rat[j]*var12_covmat[i,len(vec1)+j]-rat[i]*var12_covmat[len(vec1)+i,j]+rat[i]*rat[j]*var12_covmat[len(vec1)+i,len(vec1)+j])
    return rat_covmat

def chisq(datavec1, datavec2, covmat1, covmat2=None):
    if covmat2 is not None:
        covmat = covmat1 + covmat2 # normally distributed variables
    else:
        covmat = covmat1
    chisq = np.matmul(np.matmul((datavec1-datavec2).T,np.linalg.inv(covmat)),(datavec1-datavec2))
    return(chisq)

def snr_from_pte(data_vector, null_vector, covmat, chisq_data=None, nsamples=10**6):
    exceeds = np.zeros(nsamples)
    if chisq_data is None:
        chisq_data = chisq(data_vector, null_vector, covmat)
    sim = np.random.multivariate_normal(null_vector, covmat, size=nsamples)
    chi2null_list = []
    for i in range(nsamples):
        chisq_null = chisq(sim[i], null_vector, covmat)
        if chisq_null > chisq_data:
            exceeds[i] = 1
        chi2null_list.append(chisq_null)
    print("Number exceeding: ", len(np.where(exceeds == 1)[0]))
    pte = len(np.where(exceeds == 1)[0])/(float(nsamples))
    snr = np.sqrt(2.) * sp.special.erfinv(1.-pte)
    return(pte,snr, chi2null_list)


def KStest_raderrs(vals,err,mean, twosamp=False):
    '''Check of whether the data in radial bins are normally distributed
    Input is the specific values of the profile at a given radius (not specified)
    and the assumed Gaussian mean and error of the average.
    Output is the value of the KS statistic and the p-value'''
    from scipy.stats import ks_2samp, ks_1samp

    
    if twosamp:
        data = normal(loc=mean, scale=err, size=len(vals))
        stat, pval = ks_2samp(vals, data)
    else:
        stat, pval = ks_1samp(vals, partial(stats.norm.cdf, loc=mean, scale=err))
    if pval < 0.05:
        same=False
        # print(f'p-val is {pval}, so not the same distribution buddy')
    else:
        same=True
        #print(f'p-val is {pval}, so these are from the same distribution')

    return stat, pval, same



def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap