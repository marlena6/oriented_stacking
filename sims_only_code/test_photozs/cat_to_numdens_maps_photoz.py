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

loop = 20
mask = None # fullsky
catfile = "/mnt/scratch-lustre/mlokken/pkpatch/halos_fullsky_M_gt_1pt5E12.npy"
outpath = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/1pt5E12_to_1E15_msun/photoz_tests/"

sigma_z_1plusz = .05 # for maglim at z=0.4
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

obj = 'galaxies'
if obj=='halos':
    # catfile = "/mnt/scratch-lustre/mlokken/pkpatch/512Mpc_n256_nb14_nt2_nolambda_merge_formatted.npy"                                                                                                                                                                   
    catfile = "/mnt/scratch-lustre/mlokken/pkpatch/halos_fullsky_M_gt_1pt5E12.npy"
    outpath = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/1pt5E12_to_1E15_msun/"
    # outpath = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/alt_cosmo/nolambda/"                                                                                                                                                                             
    masswgt_odmap = True # always set to True for halos                                                                                                                                                                                                                   
if obj=='galaxies':
    catfile = "/mnt/scratch-lustre/mlokken/pkpatch/galaxy_catalogue.h5"
    outpath = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/mock_maglim/"
min_mass = 10**12
max_mass = 10**16
if min_mass==None and max_mass==None:
    mass_str = 'all_'
else:
    mass_str = '{:.0e}_{:.0e}_'.format(min_mass,max_mass)
print("Reading catalog.")
if obj=='halos':
    ra, dec, z, chi, mass = ws.read_halos(catfile, min_mass, max_mass)
elif obj=='galaxies':
    ra, dec, z, chi, mass = ws.read_galcat(catfile, min_mass, max_mass, satfrac=.15)
print("Catalog read.")


minz = 0.2
maxz = 0.6
inz = (z>minz) & (z<maxz)
ra, dec, z, chi, mass = ra[inz], dec[inz], z[inz], chi[inz], mass[inz]

# mess up all the redshifts in this range with the photo-z errors


nruns_local = loop // size
if rank == size-1:
    extras = loop % size
else:
    extras = 0

    
for l in range(nruns_local+extras):
    num = l + rank*nruns_local
    print("Rank ", rank, "number ", num)
    gal_photoz = np.random.normal(z, 0.013*(1+z))
    thetaphi, dists, M = csf.radec_to_thetaphi_sliced(ra, dec, gal_photoz, minz, maxz, 200, masses=mass)
    thetaphi = thetaphi[0]
    dists = dists[0]
    M = M[0]
    smth_scale = 0
    mid = (slice_min+slice_max)/2. * u.megaparsec
    # get the redshift at the middle of this slice
    z = z_at_value(cosmo.comoving_distance, mid) 
    nside = 4096
    if smth_scale > 0:
        # get the angular size (function gives arcseconds per kpc, convert to
        # Mpc, then multiply by user-input scale [in Mpc]
        smth_scale_arcsec = cosmo.arcsec_per_kpc_comoving(z).to(u.arcsec/u.megaparsec)*smth_scale
        odmap = csf.get_od_map(nside, thetaphi[:,0], thetaphi[:,1], mask, smth, mass=M)
        outfile = "odmap_distMpc_{:d}_{:d}_{:d}Mpc_photoz{:d}_{:d}arcmin.fits".format(int(slice_min), int(slice_max), int(smth_scale), num, int(smth_scale_arcsec.value//60.))
        print("Writing map to %s" %outpath+outfile)
        
    elif smth_scale == 0:
        smth_scale_arcsec = 0*u.arcsec
        odmap = csf.get_od_map(nside, thetaphi[:,0], thetaphi[:,1], mask, 0, mass=M)
        outfile = "odmap_distMpc_{:d}_{:d}_{:d}Mpc_photoz{:d}_{:d}arcmin.fits".format(int(slice_min), int(slice_min), int(smth_scale), num, int(smth_scale_arcsec.value//60.))
        print("Writing map to %s" %outpath+outfile)                 
    hp.write_map(outpath+outfile, odmap, overwrite=True)
