import numpy as np
import error_analysis_funcs as ef
import os
from astropy.io import fits
import coop_setup_funcs as csf
import healpy as hp

#mode = 'Websky'
# mode = 'ACTxDES'
# mode = 'Buzzard'
mode = 'Cardinal'
if mode == 'Websky':
    # cut  = 'lambda'
    cut = 'mass'
else:
    cut = 'lambda'
nreg = 24
if cut == 'lambda':
    cutmin = 20
elif cut == 'mass':
    cutmin = 1*10**13
    cutmax = 5*10**13
    #cutmax = None

if mode == 'ACTxDES': 
    # object_path = "/mnt/raid-cita/mlokken/data/cluster_cats/redmapper2.2.1_lgt20vl50_mask_actshr1deg_des_cutpt8.fit"
    object_path = "/mnt/raid-cita/mlokken/data/cluster_cats/redmapper2.2.1_lgt5vl50_mask_actshr1deg_des_cutpt8.fit"

if mode == 'Buzzard': 
    object_path = "/mnt/raid-cita/mlokken/buzzard/catalogs/combined_actdes_mask_pt8_buzzard_1.9.9_3y3a_rsshift_run_redmapper_v0.5.1_lgt05_vl50_catalog.fit"

if mode == 'Cardinal': 
    object_path = "/mnt/raid-cita/mlokken/cardinal/maglim_mask_Cardinal-3Y6a_v2.0_run_run_redmapper_v0.8.1_lgt20_vl50_catalog.fit"

if mode == 'Websky':
    object_path = "/mnt/scratch-lustre/mlokken/pkpatch/halos_fullsky_M_gt_1E13.npy"

print("Loading data.")
if mode == 'Websky' and cut == 'mass':
    ra,dec,z,cl_id = csf.get_radecz(object_path, min_mass=cutmin, max_mass = cutmax, return_id=True)
else:
    ra,dec,z,richness,cl_id = csf.get_radeczlambda(object_path, return_id=True)
    # limit with richness
    print("Cutting by richness.")
    rich_cond = richness > cutmin
    ra,dec,z,cl_id  = ra[rich_cond], dec[rich_cond], z[rich_cond], cl_id[rich_cond]

# get regions
if mode == "Websky":
    labels = (hp.ang2pix(2, ra, dec, lonlat=True)).astype(int)
else:
    labels = ef.make_regions(ra, dec, nreg, plot=True, plotroot=mode+"_{:s}gt{:d}_{:d}reg".format(cut, cutmin, nreg), mode=mode)

nreg = max(labels)+1
# save regions
if cut == 'mass':
    if cutmin is not None:
        cutminstr = 'gt{:.0e}'.format(cutmin)
    else:
        cutminstr = ''
    if cutmax is not None:
        cutmaxstr = 'lt{:.0e}'.format(cutmax)
    else:
        cutmaxstr = ''
else:
    cutminstr = 'gt{:d}'.format(cutmin)
    cutmaxstr = ''

cutstr = '{:s}'.format(cut)+cutminstr+cutmaxstr

np.savetxt("./labels_{:d}_regions_{:s}_{:s}.txt".format(nreg,mode,cutstr), np.concatenate((np.array([cl_id]).T, np.array([labels]).T), axis=1))
