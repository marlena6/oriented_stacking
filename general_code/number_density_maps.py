# Takes theta,phi lists for clusters and galaxies, plots them, and smooths them
import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import sys
from mpi4py import MPI
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u
import coop_setup_funcs as csf

#user message                                                                   
if len(sys.argv)!=8:
    sys.exit("USAGE: <path to catalog of Healpix (theta,phi) > <path to output directory> <outroot (e.g. ndmap, odmap)>  <smoothing scale [Mpc] (can be 0)> <size of slices [cmpc]> <path to mask> <mode: odmap or ndmap>\n")

path = sys.argv[1]
outpath = sys.argv[2]
outroot = sys.argv[3]
smth_scale = float(sys.argv[4])
slice_width = float(sys.argv[5])
mask_path   = sys.argv[6]
mode = sys.argv[7]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

tp_files = np.asarray([file for file in os.listdir(path) if file.startswith("thetaphi")])
mask = hp.read_map(mask_path)

fileid = []
for file in tp_files:
    root = os.path.splitext(file)[0]
    root_split = root.split("_")
    fileid.append(int(root_split[len(root_split)-1]))
# fileid is the upper distance bound of the slice
fileid = np.asarray(fileid)

sort = np.argsort(fileid)
sorted_dist = fileid[sort]
tp_files = tp_files[sort]


nmaps = len(tp_files)


nmaps_local = nmaps // size
if rank == size-1:
    extras = nmaps % size
else:
    extras = 0
for i in range(nmaps_local + extras):
    filenum    = i + nmaps_local * rank
    print("Loading file: %s" %tp_files[filenum])
    longroot   = os.path.splitext(tp_files[filenum])[0]
    shortroot  = longroot.replace("thetaphi_","")
    upperbound = sorted_dist[filenum]
    mid        = (upperbound + (upperbound-slice_width))/2. * u.megaparsec
    # get the redshift at the middle of this slice
    z = z_at_value(cosmo.comoving_distance, mid)
    tp = np.loadtxt(path + tp_files[filenum])
    theta = tp[:,0]
    phi = tp[:,1]
    nside = 4096
    if smth_scale > 0:
     # get the angular size (function gives arcseconds per kpc, convert to
        # Mpc, then multiply by user-input scale [in Mpc]
        smth_scale_arcsec = cosmo.arcsec_per_kpc_comoving(z).to(u.arcsec/u.megaparsec)*smth_scale
        if mode == 'odmap':
            newmap = csf.get_od_map(nside, theta, phi, mask, smth_scale_arcsec)
        elif mode == 'ndmap':
            newmap = csf.get_nd_map(nside, theta, phi, mask, smth_scale_arcsec)
        outfile = outroot+"_"+shortroot+"_%dMpc_%darcmin.fits" %(int(smth_scale),int(smth_scale_arcsec.value//60.))
    
    elif smth_scale == 0:
        if mode == 'odmap':
            newmap = csf.get_od_map(nside, theta,phi, mask, 0)
        elif mode == 'ndmap':
            newmap = csf.get_nd_map(nside, theta,phi, mask, 0)
        outfile = outroot+"_"+shortroot+"_0Mpc_0arcmin.fits"
    print("Writing map to %s" %outpath+outfile)
    hp.write_map(outpath+outfile, newmap, overwrite=True)
