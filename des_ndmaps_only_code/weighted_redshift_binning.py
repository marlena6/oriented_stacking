# Takes in DES catalog, comoving size you want each redshift bin to span over
# Returns same catalog with additional parameters: the weight of each cluster in each redhift bin.
# Usage: weighted_redshift_binning.py <path to DES catalog> <distance in comoving Mpc>

import numpy as np
import healpy as hp
from astropy.io import fits
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo, z_at_value
import matplotlib.pyplot as plt
import sys
from astropy.table import Column, Table

# define functions
def add_weight(col_idx, cluster_idx, weight):
    #header_index is the number index for the header parameter 
    keyword = cl_table.colnames[col_idx]
    cl_table[cluster_idx][keyword]=weight

# take inputs
if len(sys.argv)!=3:
    sys.exit("USAGE: <path to DES catalog> <Length of desired redshift bins [comoving Mpc]> \n")
des_catalog = sys.argv[1]
dist = float(sys.argv[2]) * u.megaparsec


des = fits.open(des_catalog)
desdata = des[1].data
deshdr = des[1].header
des.close()

# important parameters of catalog
ra = desdata['RA']
dec = desdata['DEC']
z = desdata['Z_LAMBDA']
z_e = desdata['Z_LAMBDA_E']
pzbins = desdata['PZBINS'] #redshifts at which we evaluate the redshift probability distribution P(z) for each cluster.
pz = desdata['PZ'] #the corresponding P(z) values

minz = min(z)
maxz = max(z)
print("Minimum redshift: %f, maximum redshift: %f" %(minz,maxz) )

# make a list called bins that will be:
# each row: [zmin, zmax],[index of clusters] for every redshift bin
bins = []
for i in range(100):
    d_slice_min = dist*i
    d_slice_max = dist*(i+1)
    z_slice_max = z_at_value(cosmo.comoving_distance, d_slice_max)
    if i==0:
        z_slice_min = 0
    else:
        z_slice_min = z_at_value(cosmo.comoving_distance, d_slice_min)
    z_mean = (z_slice_max+z_slice_min)/2.
    idx, = np.where(np.logical_and(z>z_slice_min, z<z_slice_max)) #indices of clusters whose mean photo-z is in this redshift slice
    bins.append([[z_slice_min, z_slice_max],idx])
    if(z_slice_max)>maxz:
        break
print("Number of bins: %d"%len(bins))
# Add these bins to the table of cluster information

cl_table = Table(desdata)
data = np.zeros(len(cl_table))
ncols = len(cl_table.colnames)
for i in range(len(bins)):
    zmin,zmax = bins[i][0]
    name = '%1.2f_to_%1.2f'%(zmin,zmax)
    col = Column(data=data, name=name, dtype=float)
    cl_table.add_column(col)

for b in range(len(bins)):
    bin_min = bins[b][0][0]
    bin_max = bins[b][0][1]
    idx = bins[b][1]
    for i in idx: # for each cluster in this bin
        cluster = desdata[i]
        photoz_min = pzbins[i][0]
        photoz_max = pzbins[i][len(pzbins[i])-1]
        # If the cluster photo-z range falls completely in this bin:
        if photoz_min > bin_min and photoz_max < bin_max:
            # Weight this bin with a 1 in the catalog:
            add_weight(ncols+b, i, 1.)
        elif photoz_min>bin_min and photoz_max > bin_max: # if the cluster photo-z range is within the bin on the lower end but outside of it on the upper end
            #we will do an integral over the mean bin
            integral1 = 0
            #then we'll integrate over the bin above it (just 1)
            integral2 = 0
            binsize = (photoz_max - photoz_min)/len(pzbins[i])
            # for every probability value in the photoz probability distribution,
            for m in range(len(pz[i])):   
                val = pz[i][m] # take that value
                if pzbins[i][m]<bin_max:
                    integral1 += val * binsize # multiply it by the distance between bins
                elif pzbins[i][m]>bin_max and pzbins[i][m]<bins[b+1][0][1]: #if we're within the next bin up
                    integral2 += val * binsize
            add_weight(ncols+b, i, integral1)
            add_weight(ncols+b+1, i, integral2)
        elif photoz_min<bin_min and photoz_max < bin_max: #if the cluster photo-z range is within the bin on the higher end but outside on the lower end
            integral1 = 0
            integral2 = 0
            binsize = (photoz_max-photoz_min)/len(pzbins[i])
            for m in range(len(pz[i])-1,-1,-1):
                val = pz[i][m]
                if pzbins[i][m]>bin_min:
                    integral1+=val*binsize
                elif pzbins[i][m]<bin_min:
                    integral2+=val*binsize
            add_weight(ncols+b, i, integral1)
            add_weight(ncols++b-1, i, integral2)
        elif photoz_min<bin_min and photoz_max > bin_max:
            binsize = (photoz_max-photoz_min)/len(pzbins[i])
            integrallow = 0
            integralmid = 0
            integralhigh = 0
            for m in range(len(pz[i])):
                val = pz[i][m]
                if pzbins[i][m]<bin_min:
                   integrallow += val*binsize
                elif pzbins[i][m]>bin_min and pzbins[i][m]<bin_max:
                    integralmid += val*binsize
                elif pzbins[i][m]>bin_max:
                    integralhigh += val*binsize
            add_weight(ncols+b, i, integralmid)
            add_weight(ncols+b-1, i, integrallow)
            add_weight(ncols+b+1, i, integralhigh)

newfile = des_catalog.strip('.fits')+'_%sMpc_bins.fits'%sys.argv[2]
cl_table.write(newfile,fits)
print("Wrote %s with redshift weights added\n"%newfile)
