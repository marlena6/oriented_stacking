import numpy as np
import error_analysis_funcs as ef
import os
from astropy.cosmology import Planck15 as cosmo, z_at_value
import astropy.units as u
from mpi4py import MPI
import subprocess
import coop_post_processing as cpp
import coop_setup_funcs as csf
from astropy.io import fits
import matplotlib.pyplot as plt
h = (cosmo.H0/100.).value
mode = 'ACTxDES'
if mode == 'Buzzard':
    object_path = "/mnt/raid-cita/mlokken/buzzard/catalogs/small_region_buzzard_1.9.9_3y3a_rsshift_run_redmapper_v0.5.1_lgt05_vl50_catalog.fit"
    pkmap_path  = "/mnt/raid-cita/mlokken/buzzard/number_density_maps/des_reg/"
    ymap        = "/mnt/raid-cita/mlokken/buzzard/ymaps/ymap_buzzard_COMBINED_NM50_Nz16_nside4096_v01_actbeam.fits"
    pkmask      = "/mnt/raid-cita/mlokken/data/redmagic/y3_gold_2.2.1_wide_sofcol_run_redmapper_v6.4.22_redmagic_highdens_0.5_vlim_zmask_hpx.fit"
    ymask       = "/mnt/raid-cita/mlokken/buzzard/ymaps/my_buzzardy_mask.fits"
    outpath     = "/mnt/scratch-lustre/mlokken/stacking/Buzzard_tSZ_rmpks/"
    modestr = 'buzzard_'
if mode == 'ACTxDES':
    pkmap_path = "/mnt/raid-cita/mlokken/data/number_density_maps/"
    object_path = "/mnt/raid-cita/mlokken/data/cluster_cats/y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_lgt5_vl50_catalog.fit"
    ymap        = "/mnt/raid-cita/mlokken/data/act_ymaps/tilec_single_tile_deep56_comptony_map_v1.2.0_joint_4096_hpx.fits"
    pkmask      = "/mnt/raid-cita/mlokken/data/redmagic/y3_gold_2.2.1_wide_sofcol_run_redmapper_v6.4.22_redmagic_highdens_0.5_vlim_zmask_hpx.fit"
    ymask       = "/mnt/raid-cita/mlokken/data/act_ymaps/tilec_mask_4096_hpx.fits"
    outpath     = "/mnt/scratch-lustre/mlokken/stacking/ACT+Planck/"
    modestr     = ''
zbins =[[0.15, 0.35], [0.35, 0.5], [0.5, 0.65]]
dbins = [[cosmo.comoving_distance(z[0]).value,cosmo.comoving_distance(z[1]).value] for z in zbins]

pt_selection_str = "allnu_alle_"
e_min  = 0.
e_max  = 1000.
nu_min = -100

width    = 200
smth     = 9.4/h
smth_str = ("{:.1f}".format(smth)).replace('.','pt')
minz     = 0.1
cut      = 'lambda'
cutmin   = 10

cutstr = '{:s}gt{:d}'.format(cut,cutmin)
standard_stk_file = "standard_stackfile.ini"
standard_pk_file  = "standard_pkfile.ini"

ra,dec,z,richness = csf.get_radeczlambda(object_path)
# limit with richness                                                                                                                          
print("Cutting by richness.")
rich_cond = richness > cutmin
ra,dec,z  = ra[rich_cond], dec[rich_cond], z[rich_cond]


thetaphi_list,dlist = csf.radec_to_thetaphi_sliced(ra, dec, z, zbins[0][0], zbins[2][1], dbins[0][1]-dbins[0][0], zbins=zbins)
print(dlist)
c = 0
for zbin in zbins:
    dlow, dhi = int(dbins[c][0]), int(dbins[c][1])
    print("{:d} to {:d} Mpc\n".format(dlow, dhi))
    if mode == 'Buzzard' or mode == 'ACTxDES':
        tp_regfile = os.path.join(outpath, "thetaphi_fullmap_{:d}_{:d}Mpc_{:s}.txt".format(dlow, dhi, cutstr))
        np.savetxt(tp_regfile, thetaphi_list[c])
        pkmap = os.path.join(pkmap_path, "3_zbins_grf_comparison", "{:s}redmagic_highdens_zbin{:d}_od_map_nosmooth.fits".format(modestr,c+1))
        inifile_root = "{:s}_{:s}_fullmap_{:d}_{:d}Mpc_{:s}{:s}".format(mode, cutstr, dlow, dhi, pt_selection_str, smth_str)
    pk_ini, stk_ini = ef.make_ini_files(pkmap, ymap,  tp_regfile, smth, standard_pk_file, standard_stk_file, outpath, inifile_root, [dlow,dhi], pk_mask=pkmask, stk_mask=ymask, e_min=e_min, e_max=e_max, nu_min=nu_min)
    print("Running GetPeaks on {:s}".format(pk_ini))
    subprocess.run(args=["/home/mlokken/software/COOP/mapio/GetPeaks",pk_ini])
    #print("Running Stack on {:s}".format(stk_ini))
    #subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",stk_ini])
    c += 1
