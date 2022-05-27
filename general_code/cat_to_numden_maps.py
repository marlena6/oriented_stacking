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

# inputs for different options
mode   = "peakpatch"
split  = False
masswgt_odmap = False
smth_scale    = 0 * u.Mpc
#45 * u.Mpc

if (smth_scale == 0*u.Mpc) or (smth_scale==0):
    smth_str = ''
else:
    smth_str = '_smth_'+str(int(smth_scale.value))+'Mpc'
if mode == "peakpatch":
    mask_path = None # fullsky
    # mask_path = "/mnt/raid-cita/mlokken/data/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_hpx_4096.fits"
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

elif mode == "maglim":
    mask_path = "/mnt/raid-cita/mlokken/data/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_hpx_4096.fits"
    catfile   = "/mnt/raid-cita/mlokken/data/maglim/maglim_data_wflux_wmag.fits"
    outpath   = "/mnt/raid-cita/mlokken/data/number_density_maps/maglim/"
    mass_str  = ''
    with fits.open(catfile) as cat:
        ra, dec, z, mass = cat[1].data['ra'], cat[1].data['dec'], cat[1].data['z_mean'], np.zeros(len(cat[1].data))
elif mode == "maglim_buzz":
    print("Maglim Buzzard.")
    mask_path = "/mnt/raid-cita/mlokken/data/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_hpx_4096.fits"
    catpath   = "/mnt/raid-cita/mlokken/buzzard/catalogs/"
    outpath   = "/mnt/raid-cita/mlokken/buzzard/number_density_maps/maglim/"
    mass_str  = ''
    ra   = []
    dec  = []
    z    = []
    mass = []
    for catfile in os.listdir(catpath):
        if catfile.startswith("maglim_buzz_Ndensity"):
            print(catfile)
            with fits.open(os.path.join(catpath,catfile)) as cat:
                ra.extend(cat[1].data['ra'])
                dec.extend(cat[1].data['dec'])
                z.extend(cat[1].data['DNF_ZMEAN'])
                mass.extend(np.zeros(len(cat[1].data)))
    ra  = np.asarray(ra)
    dec = np.asarray(dec)
    z   = np.asarray(z)
    mass = np.asarray(mass)
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
    minz  = 0.3
    maxz  = 0.5
nside = 4096

odmap_frac = .75
pct = str(int(odmap_frac*100))

if mask_path is not None:
    mask = hp.read_map(mask_path)
else:
    mask = None
if split:
    half1     = np.zeros(len(ra),dtype=bool)
    idx_half1 = np.random.choice(np.arange(len(half1)), size=int(len(ra)*odmap_frac), replace=False)
    half1[idx_half1] = True
    half2     = np.logical_not(half1)

    c = 1
    for idx in [half1, half2]:
        if zsplit:
            thetaphi, dists, M = csf.radec_to_thetaphi_sliced(ra[idx], dec[idx], z[idx], zbins=zbins, masses=mass[idx])
        else:
            thetaphi, dists, M = csf.radec_to_thetaphi_sliced(ra[idx], dec[idx], z[idx], minz, maxz, 500, masses=mass[idx]) # constant-comoving-distance method
        nruns_local = len(thetaphi) // size
        print(len(ra[idx])/len(ra))
        if rank == size-1:
            extras = len(thetaphi) % size
        else:
            extras = 0

        for i in range(nruns_local+extras):
            n = i + nruns_local*rank
            distbin = dists[n]
            print("Getting overdensity and number density maps for bin ", distbin)

            if zsplit:
                zbin = zbins[n]
            thetaphi_list = thetaphi[n]
            M_inbin = M[n]
            if c==1: # get the overdensity map which will be used for orientation
                if masswgt_odmap:
                    map = csf.get_od_map(nside, thetaphi_list[:,0], thetaphi_list[:,1], mask, 0, mass=M_inbin)
                else:
                    map = csf.get_od_map(nside, thetaphi_list[:,0], thetaphi_list[:,1], mask, 0)
                label = 'odmap'
                pctlabel = pct
            elif c==2: # get the number density map which will be used for orientation
                map = csf.get_nd_map(nside, thetaphi_list[:,0], thetaphi_list[:,1], mask, 0)
                label = 'ndmap'
                pctlabel = str(int((1-odmap_frac)*100))
            if zsplit:
                outfile = "{:s}_{:s}_{:s}z_{:s}_{:s}.fits".format(label, pctlabel, mass_str, str(zbin[0]).replace('.','pt'), str(zbin[1]).replace('.','pt'))
            else:
                outfile = "{:s}_{:s}_{:s}{:d}_{:d}Mpc.fits".format(label, pctlabel, mass_str, distbin[0], distbin[1])
            print("Writing map to %s" %outpath+outfile)
            hp.write_map(outpath+outfile, map, overwrite=True)
        c+=1

else:
    if zsplit:
        thetaphi, dists, M = csf.radec_to_thetaphi_sliced(ra, dec, z, zbins=zbins, masses=mass)
    else:
        thetaphi, dists, M = csf.radec_to_thetaphi_sliced(ra, dec, z, minz, maxz, 50, masses=mass) # constant-comoving-distance method

    nruns_local = len(thetaphi) // size

    if rank == size-1:
        extras = len(thetaphi) % size
    else:
        extras = 0

    for i in range(nruns_local+extras):
        n = i + nruns_local*rank
        distbin = dists[n]
        if zsplit:
            zbin = zbins[n]
            print("Making map for z range ", zbin)
        print("Comoving distance range: ", distbin)
        thetaphi_list = thetaphi[n]
        M_inbin = M[n]
        if zsplit:
            mid = np.average([cosmo.comoving_distance(zbin[0]).value,cosmo.comoving_distance(zbin[1]).value])*u.Mpc
        else:
            mid = (distbin[0]+distbin[1])/2. * u.megaparsec
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
        if masswgt_odmap:
            print("Weighting the overdensity map by mass.")
            odmap = csf.get_od_map(nside, thetaphi_list[:,0], thetaphi_list[:,1], mask, smth_scale_arcsec.value, mass=M_inbin)
        else:
            odmap = csf.get_od_map(nside, thetaphi_list[:,0], thetaphi_list[:,1], mask, smth_scale_arcsec.value)
        if zsplit:
            outfile = "odmap_{:s}z_{:s}_{:s}{:s}.fits".format(mass_str, str(zbin[0]).replace('.','pt'), str(zbin[1]).replace('.','pt'), smth_str)
        else:
            outfile = "odmap_{:s}distMpc{:d}_{:d}{:s}.fits".format(mass_str, distbin[0], distbin[1], smth_str)
        print("Writing map to %s" %outpath+outfile)
        hp.write_map(outpath+outfile, odmap, overwrite=False)


        #            outfile = "odmap_distMpc_{:d}_{:d}_{:d}Mpc_{:d}arcmin.fits".format(int(distbin[0]), int(distbin[1]), int(smth_scale), int(smth_scale_arcsec.value//60.))
