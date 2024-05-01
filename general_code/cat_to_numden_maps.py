# Takes theta,phi lists for clusters and galaxies, plots them, and smooths them

import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import sys
from mpi4py import MPI
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u
import coop_setup_funcs as csf
import websky as ws
from astropy.io import fits
import h5py

# inputs for different options
# mode   = "peakpatch"
# mode = "redmagic"
# mode = "maglim"
#mode = "redmagic_buzz"
# mode = "maglim_buzz"
mode = "maglim_cardinal"

split = True
masswgt_odmap = False
smth_scale    = 0 * u.Mpc
#45 * u.Mpc



if mode == "peakpatch":
    mask_path = None # fullsky
    # mask_path = "/mnt/raid-cita/mlokken/data/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_hpx_4096.fits"
    obj = 'halos'
    if obj=='halos':
        # catfile = "/mnt/scratch-lustre/mlokken/pkpatch/512Mpc_n256_nb14_nt2_nolambda_merge_formatted.npy"
        catfile = "/mnt/raid-cita/mlokken/pkpatch/halos_fullsky_M_gt_1pt5E12.npy"
        outpath = "/mnt/raid-cita/mlokken/pkpatch/number_density_maps/fullsky/1pt5E12_to_1E15_msun/"
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
    if masswgt_odmap:
        w = mass/(10**12)
    else:
        w = 1

elif mode == "maglim":
    mask_path = "/mnt/raid-cita/mlokken/data/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_hpx_4096.fits"
    fracmask_path = "/mnt/raid-cita/mlokken/data/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_fracgood_hpx_4096.fits"
    catfile   = "/mnt/raid-cita/mlokken/data/maglim/maglim_data_wflux_wmag.fits"
    outpath   = "/mnt/raid-cita/mlokken/data/number_density_maps/maglim/"
    mass_str  = ''
    with fits.open(catfile) as cat:
        catlen = len(cat[1].data)
        ra, dec, z, w = cat[1].data['ra'], cat[1].data['dec'], cat[1].data['z_mean'], cat[1].data['weight']
elif mode == "redmagic":
    mask_path = "/mnt/raid-cita/mlokken/data/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_hpx_4096.fits"
    catfile   = "/mnt/raid-cita/mlokken/data/redmagic/redmagic_broad_chi2_isdw_subsamp_chi2_8.fits"
    outpath   = "/mnt/raid-cita/mlokken/data/number_density_maps/200_cmpc_slices/redmagic_updated_nov2021/"
    mass_str  = ''
    with fits.open(catfile) as cat:
        ra, dec, z, w = cat[1].data['ra'], cat[1].data['dec'], cat[1].data['z'], cat[1].data['w']
    
elif mode == "maglim_buzz":
    print("Maglim Buzzard.")
    mask_path = "/mnt/raid-cita/mlokken/data/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_hpx_4096.fits" # binary
    fracmask_path = "/mnt/raid-cita/mlokken/data/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_fracgood_hpx_4096.fits"
    catpath   = "/mnt/raid-cita/mlokken/buzzard/catalogs/"
    outpath   = "/mnt/raid-cita/mlokken/buzzard/number_density_maps/maglim/"
    
    mass_str  = ''
    ra   = []
    dec  = []
    z    = []
    catlen = 0
    for catfile in os.listdir(catpath):
        if catfile.startswith("maglim_buzz_Ndensity"):
            print(catfile)
            with fits.open(os.path.join(catpath,catfile)) as cat:
                ra.extend(cat[1].data['ra'])
                dec.extend(cat[1].data['dec'])
                z.extend(cat[1].data['DNF_ZMEAN'])
                catlen+= len(cat[1].data)
    ra  = np.asarray(ra)
    dec = np.asarray(dec)
    z   = np.asarray(z)
    w   = 1
    
elif mode == "maglim_cardinal":
    print("Maglim Cardinal.")
    mask_path = "/mnt/raid-cita/mlokken/cardinal/cardinal_maglim_mask.fits"
    fracmask_path = "/mnt/raid-cita/mlokken/cardinal/fracgood_band_i_nside_4096.hp"
    catfile   = "/mnt/raid-cita/mlokken/cardinal/maglim_Cardinal-3_v2.0_Y6a.hdf5"
    outpath   = "/mnt/raid-cita/mlokken/cardinal/number_density_maps/maglim/"
    mass_str  = ''
    
    with h5py.File(catfile, 'r') as cat:
        ra, dec, z, w = cat['ra'][:], cat['dec'][:], cat['DNF_ZMEAN'][:], cat['weight'][:]
        catlen = len(ra)
    cat.close()
    
elif mode == "redmagic_buzz":
    print("RedMaGiC Buzzard.")
    mask_path = "/mnt/raid-cita/mlokken/data/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_hpx_4096.fits"
    catfile   = "/mnt/raid-cita/mlokken/buzzard/catalogs/buzzard_1.9.9_3y3a_rsshift_run_redmagic_highdens.fit"
    outpath   = "/mnt/raid-cita/mlokken/buzzard/number_density_maps/redmagic/"
    mass_str  = ''
    with fits.open(catfile) as cat:
        ra, dec, z = cat[1].data['ra'], cat[1].data['dec'], cat[1].data['zredmagic']
    w   = 1


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

    
zsplit = False # divide by redshift bins? if false, will divide by comoving distace bins
# Maglim zbins
if zsplit:
    zbins = [[0.20, 0.40], [0.40,0.55], [0.55,0.70], [0.70,0.85], [0.85,0.95], [0.95,1.05]]
    #zbins = [[0.55,0.70]]
# no-lambda test run only goes out to ~.2
else:
    # minz  = 0.4
    # maxz  = 0.47
    minz  = 0.2
    maxz  = 1.05 
nside = 4096

if split:
    odmap_frac = .75
    pct = str(int(odmap_frac*100))

overlap = True

if mask_path is not None:
    mask = hp.read_map(mask_path)
else:
    mask = None
if fracmask_path is not None:
    fra = hp.read_map(fracmask_path)


if split:
    from numpy.random import Generator, PCG64
    rng = Generator(PCG64(12345)) # set the seed to be consistent every time
    frac1     = np.zeros(len(ra),dtype=bool)
    idx_frac1 = rng.choice(np.arange(len(ra)), size=int(len(ra)*odmap_frac), replace=False)
    frac1[idx_frac1] = True
    frac2     = np.logical_not(frac1)
    ra2 = ra[frac2]
    dec2= dec[frac2]
    z2  = z[frac2]
    ra = ra[frac1]
    dec= dec[frac1]
    z  = z[frac1]
    if type(w)!=int:
        w2 = w[frac2]
        w = w[frac1]
    print(f"Total sample has {catlen} galaxies. Split the data into sample of length {len(ra)} (fraction = {len(ra)/catlen} for orientation and sample of length {len(ra2)} (fraction = {len(ra2)/catlen}) for stacking.")
if zsplit:
    thetaphi_list, distlist, binlist = csf.radec_to_thetaphi_sliced(ra, dec, z, zbins=zbins)
else:
    thetaphi_list, distlist, binlist = csf.radec_to_thetaphi_sliced(ra, dec, z, minz, maxz, 200) # constant-comoving-distance method
    if overlap:
        thetaphi_list_off, distlist_off, binlist_off = csf.radec_to_thetaphi_sliced(ra, dec, z, minz, maxz, 200, offset=100)
        thetaphi_list = thetaphi_list+thetaphi_list_off
        binlist  = binlist + binlist_off
        distlist = distlist+distlist_off
if type(w)!=int:
    wlist = [w[bin] for bin in binlist]
    
# set the number of distance bins each core will run through
nruns_local = len(thetaphi_list) // size
if rank == size-1:
    extras = len(thetaphi_list) % size
else:
    extras = 0

# run through the distance bins
for i in range(nruns_local+extras):
    if (smth_scale == 0*u.Mpc) or (smth_scale==0):
        smth_str = ''
    else:
        smth_str = '_smth_'+str(int(smth_scale.value))+'Mpc'
    n = i + nruns_local*rank
    distbin = distlist[n]
    thetaphi = thetaphi_list[n]
    if type(w)!=int:
        weight = wlist[n]
    else:
        weight = w
    print("Rank {:d}: Getting overdensity and number density maps for bin ".format(rank), distbin)
    if split:
        label    = 'odmap'
        pctlabel = pct
        if zsplit:
            zbin = zbins[n]
            outfile = "{:s}_{:s}_{:s}z_{:s}_{:s}_cc.fits".format(label, pctlabel, mass_str, str(zbin[0]).replace('.','pt'), str(zbin[1]).replace('.','pt'))
        else:
            outfile = "{:s}_{:s}_{:s}{:d}_{:d}Mpc_cc.fits".format(label, pctlabel, mass_str, distbin[0], distbin[1])
        if os.path.exists(os.path.join(outpath,outfile)):
            print("Map already made. Moving on.\n")
        else:
            # get the overdensity map which will be used for orientation
            map = csf.get_od_map(nside, thetaphi[:,0], thetaphi[:,1], binmask=mask,fracmask=fra, smth=0, wgt=weight)
            print("Writing map to %s" %outpath+outfile)
            hp.write_map(outpath+outfile, map, overwrite=True)
            
    else:
        mid = (distbin[0]+distbin[1])/2.*u.Mpc
        print("Middle of slice comoving distance ", mid)
        # get the redshift at the middle of this slice
        z_mid = z_at_value(cosmo.comoving_distance, mid)
        print("Corresponding z: ", z_mid)
        if smth_scale > 0:
            # get the angular size (function gives arcseconds per kpc, convert to
            # Mpc, then multiply by user-input scale [in Mpc]
            smth_scale_arcsec = cosmo.arcsec_per_kpc_comoving(z_mid).to(u.arcsec/u.megaparsec)*smth_scale
            smth_str += '_'+str(int(smth_scale_arcsec.value//60.))+'a'
        else:
            smth_scale_arcsec = 0*u.arcsec
        label = 'odmap'
        if zsplit:
            zbin = zbins[n]
            outfile = "{:s}_100_{:s}z_{:s}_{:s}{:s}_cc.fits".format(label, mass_str, str(zbin[0]).replace('.','pt'), str(zbin[1]).replace('.','pt'), smth_str)
        else:
            outfile = "{:s}_100_{:s}{:d}_{:d}Mpc{:s}_cc.fits".format(label, mass_str, distbin[0], distbin[1], smth_str)
        if os.path.exists(os.path.join(outpath,outfile)):
               print("Map already made. Moving on.\n")
        else:
            map = csf.get_od_map(nside, thetaphi[:,0], thetaphi[:,1],  binmask=mask,fracmask=fra, smth=smth_scale_arcsec.value, wgt=weight)
            print("Writing map to %s" %outpath+outfile)
            hp.write_map(outpath+outfile, map, overwrite=True)
if split:
    maptype = 'nd' # maptype for the second map to stack
    # make the maps-to-stack out of the rest
    zbins = [[0.19985555905328484, 0.3565167560560754],[0.3565167560560754, 0.5289988643902372],[0.5289988643902372, 0.7215854982816572],
             [0.7215854982816572, 0.9396687416637612]]
    thetaphi_list, distlist, binlist = csf.radec_to_thetaphi_sliced(ra2, dec2, z2, zbins=zbins)
    nruns_local2 = len(thetaphi_list) // size
    if rank == size-1:
        extras = len(thetaphi_list) % size
    else:
        extras = 0
    
    if type(w)!=int:
        wlist = [w[bin] for bin in binlist]    
    for i in range(nruns_local2+extras):
        n = i + nruns_local2*rank

        distbin = distlist[n]
        thetaphi = thetaphi_list[n]
        if type(w)!=int:
            weight = wlist[n]
        else:
            weight = w
        zbin = zbins[n]
        pctlabel=str(int((1-odmap_frac)*100))
        outfile = "{:s}map_{:s}_{:s}z_{:.2f}_{:.2f}{:s}_cc.fits".format(maptype, pctlabel, mass_str, zbin[0], zbin[1], smth_str).replace(".","pt",2)
        if os.path.exists(outpath+outfile):
            print('Map already made. Moving on.')
        else:
            if maptype=='nd':
                map = csf.get_nd_map(nside, thetaphi[:,0], thetaphi[:,1],  binmask=mask,fracmask=fra, smth=0, wgt=weight)
            elif maptype=='nu':
                map = csf.get_nu_map(nside, thetaphi[:,0], thetaphi[:,1], binmask=mask,fracmask=fra, smth=0, wgt=weight)
            else:
                sys.exit("map must be either nd or nu.  Exiting.")
            print("Writing map to %s" %outpath+outfile)
            hp.write_map(outpath+outfile, map, overwrite=True)
