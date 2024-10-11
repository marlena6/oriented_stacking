import numpy as np
import error_analysis_funcs as ef
import os
from astropy.cosmology import Planck18 as cosmo, z_at_value
import astropy.units as u
import subprocess
import coop_post_processing as cpp
import coop_setup_funcs as csf
from astropy.io import fits
import matplotlib.pyplot as plt

standard_stk_file = "standard_stackfile.ini"
standard_pk_file  = "standard_pkfile.ini"

mode = 'maglim'
object_path = "/mnt/raid-cita/mlokken/data/cluster_cats/small_region_y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_lgt5_vl50_catalog.fit"
pkmap_path  = "/mnt/raid-cita/mlokken/data/number_density_maps/maglim/"
ymap        = "/mnt/raid-cita/mlokken/data/act_ymaps/ilc_fullRes_SZ_yy_4096_hpx.fits"
pkmask      = "/mnt/raid-cita/mlokken/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_hpx_4096.fits"
ymask       = "/mnt/raid-cita/mlokken/data/act_ymaps/BN_bottomcut_increased_apod_v2.fits"
outpath     = "/mnt/scratch-lustre/mlokken/stacking/maglim_tests/"

ra,dec,z,richness = csf.get_radeczlambda(object_path)
# limit with richness
print("Cutting by richness.")
lmin = 10
cutstr = 'lgt{:d}'.format(lmin)
rich_cond = richness > lmin
smth = 10
ra,dec,z  = ra[rich_cond], dec[rich_cond], z[rich_cond]

thetaphi_list,dlist = csf.radec_to_thetaphi_sliced(ra, dec, z, zbins=[[0.2,0.40]])

for i in range(len(thetaphi_list)):
    dlow, dhi = dlist[i][0], dlist[i][1]
    tp_regfile = os.path.join(outpath, "thetaphi_bin1_{:s}.txt".format(cutstr))
    np.savetxt(tp_regfile, thetaphi_list[i])
    for pct in ['25pct','50pct','75pct', '']:
        pkmap = os.path.join(pkmap_path, "odmap{:s}_bin1.fits".format(pct))
        tp_regfile = os.path.join(outpath, "thetaphi_bin1_{:s}.txt".format(cutstr))
    # make the ini files
        inifile_root = "{:s}_{:s}_bin1{:s}_smth{:d}Mpc".format(mode, cutstr, pct, smth)
        pk_ini, stk_ini = ef.make_ini_files(pkmap, ymap, tp_regfile, smth, standard_pk_file, standard_stk_file, outpath, inifile_root, [dlow,dhi], pk_mask=pkmask, stk_mask=ymask)
        print("Running GetPeaks on {:s}".format(pk_ini))
        subprocess.run(args=["/home/mlokken/software/COOP_new/mapio/GetPeaks",pk_ini])
    #print("Running Stack on {:s}".format(stk_ini))
    #subprocess.run(args=["/home/mlokken/software/COOP_new/mapio/Stack",stk_ini])
