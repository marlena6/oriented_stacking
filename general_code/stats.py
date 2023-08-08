import numpy as np
def ratio_covmat(vec1, vec2, samples1, samples2, weights1=None, weights2=None):
    # calculates the covariance matrix for a ratio of two vectors
    # samples1 and samples2 are arrays with each row corresponding to a variable,
    # each column corresponding to an observation
    rat = vec1/vec2
    if weights1 is None:
        weights1 = np.ones(samples1.shape[1])
    if weights2 is None:
        weights2 = np.ones(samples2.shape[1])
    datavec_samples = np.concatenate((samples1, samples2), axis=0) # should be an array 2*len(vec1) x len(samples1)
    var12_covmat = np.cov(datavec_samples, aweights=weights1) # covmat of var1(r1), var1(r2), etc for variable 1
    
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

def snr_from_pte(data_vector, null_vector, covmat, chisq_data=None):
    size = 1*10**6
#     size = 1*10**3
    exceeds = np.zeros(size)
    if chisq_data is None:
        chisq_data = chisq(data_vector, null_vector, covmat)
    sim = np.random.multivariate_normal(null_vector, covmat, size=size)
    for i in range(size):
        chisq_null = chisq(sim[i], null_vector, covmat)
        if chisq_null > chisq_data:
            exceeds[i] = 1
    print("Number exceeding: ", len(np.where(exceeds == 1)[0]))
    pte = len(np.where(exceeds == 1)[0])/(float(size))
    snr = np.sqrt(2.) * sp.special.erfinv(1.-pte)
    return(pte,snr)