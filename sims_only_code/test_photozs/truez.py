import numpy as np
import coop_setup_funcs as csf
from astropy.cosmology import Planck18 as cosmo, z_at_value
import astropy.units as u
from scipy.integrate import quad
from scipy.stats import norm
import matplotlib.pyplot as plt
import healpy as hp
from astropy.io import fits
import math

nside = 4096
mode  = 'buzzard'        

if mode=='buzzard':
    mask = hp.read_map("/mnt/raid-cita/mlokken/data/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_hpx_4096.fits")
elif mode=='websky':
    mask = None
 
# import catalog
if mode=='websky':
    pzcat = np.load("/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/mock_maglim/mock_maglim_photoz.npy")
    ra, dec, truez = pzcat[:,0], pzcat[:,1], pzcat[:,3]
elif mode=='buzzard':
    ra    = []
    dec   = []
    truez = []
    for i in range(6):
        with fits.open("/mnt/raid-cita/mlokken/buzzard/catalogs/maglim_buzz_Ndensity_{:d}.fits".format(i)) as buzzard:
            hdr   = buzzard[1].header
            dat   = buzzard[1].data
            ra.extend(dat['ra'])
            dec.extend(dat['dec'])
            truez.extend(dat['z'])
    ra = np.asarray(ra)
    dec = np.asarray(dec)
    truez=np.asarray(truez)
    print(ra.shape)
size = len(ra)
print(size, "galaxies.")

minz = 0.3
maxz = 0.6
# define the bins
dlist     = csf.dlist(minz=minz, maxz=maxz, slice_width=200)
dlist_off = csf.dlist(minz=minz, maxz=maxz, slice_width=200, offset=100)
# unweighted
print(truez.shape)
print(truez[:10])
print(minz, maxz)
thetaphi, dists = csf.radec_to_thetaphi_sliced(ra, dec, truez, minz, maxz, 200)
for i in range(len(dlist)):
    dbin = dlist[i]
    binmin,binmax = dbin[0], dbin[1]
    print("Bin ranging from {:d} to {:d} Mpc.".format(binmin,binmax))
    if len(thetaphi[i])!=0:
        odmap = csf.get_od_map(nside, thetaphi[i][:,0], thetaphi[i][:,1], mask, 0)
        if mode=='websky':
            outfile = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/mock_maglim/truez/odmap_mock_maglim_photoz_nowgt_distMpc{:d}_{:d}_0arcmin.fits".format(binmin, binmax)
        elif mode=='buzzard':
            outfile = "/mnt/raid-cita/mlokken/buzzard/testing_photoz/truez/odmap_mock_maglim_truez_distMpc{:d}_{:d}_0arcmin.fits".format(binmin, binmax)
        print("Writing map to %s" %outfile)
        hp.write_map(outfile, odmap, overwrite=True)

thetaphi_off, dists_off = csf.radec_to_thetaphi_sliced(ra, dec, truez, minz, maxz, 200, offset=100)
for i in range(len(dlist_off)):
    dbin = dlist_off[i]
    binmin,binmax = dbin[0], dbin[1]
    print("Bin ranging from {:d} to {:d} Mpc.".format(binmin,binmax))
    if len(thetaphi[i])!=0:
        odmap = csf.get_od_map(nside, thetaphi_off[i][:,0], thetaphi_off[i][:,1], mask, 0)
        if mode=='websky':
            outfile = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/mock_maglim/truez/odmap_mock_maglim_photoz_nowgt_distMpc{:d}_{:d}_0arcmin.fits".format(binmin, binmax)
        elif mode=='buzzard':
            outfile = "/mnt/raid-cita/mlokken/buzzard/testing_photoz/truez/odmap_mock_maglim_truez_distMpc{:d}_{:d}_0arcmin.fits".format(binmin, binmax)
        print("Writing map to %s" %outfile)
        hp.write_map(outfile, odmap, overwrite=True)

