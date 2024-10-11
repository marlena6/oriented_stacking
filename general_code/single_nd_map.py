# Takes theta,phi lists for clusters and galaxies, plots them, and smooths them                                                                                                                           
import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import sys
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u


#user message                                                                                                                                                                                             
if len(sys.argv)!=5:
    sys.exit("USAGE: <path to catalog of Healpix (theta,phi) > <path to output directory> <outroot (ndmap)>  <smoothing scale [arcmin]>\n")


path = sys.argv[1]
outpath = sys.argv[2]
outroot = sys.argv[3]
smth_scale = float(sys.argv[4])*u.arcmin


tp_files = np.asarray([file for file in os.listdir(path) if file.startswith("thetaphi")])
nfiles = len(tp_files)

nside = 4096 
halomap = np.zeros(12*nside**2)

for i in range(nfiles):
    print("Loading file: %s" %tp_files[i])
    tp = np.loadtxt(path + tp_files[i])
    theta = tp[:,0]
    phi = tp[:,1]
    nside = 4096
    pix = hp.ang2pix(nside,theta,phi)
    np.add.at(halomap, pix, 1.)

smoothed_map = hp.sphtfunc.smoothing(halomap, fwhm = np.deg2rad(smth_scale.value/60.), pol=False)
outfile = outroot+"_%sarcmin.fits"%(sys.argv[4])
print("Writing map to %s" %outpath+outfile)
hp.write_map(outpath+outfile, smoothed_map)
