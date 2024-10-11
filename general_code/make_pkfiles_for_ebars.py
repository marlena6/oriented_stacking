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
from shutil import copyfile
from astropy.table import Table
import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print("Rank = ", rank)
# the ingredients for stacking
# input a path to a catalog of points in theta,phi format -- if it is one file just do this for one file, if it is a path to many do it for all the files within
# number of regions

h = (cosmo.H0/100.).value
mode = 'ACTxDES'
nreg = 12
width    = 200
#smth     = [10,13.9,17.8]
smth = [13.9]
smth_str = [("{:.1f}".format(s)).replace('.','pt') for s in smth]
print(smth_str)
minz     = z_at_value(cosmo.comoving_distance, 1032*u.Mpc)

nu_e_cuts = True
if nu_e_cuts:
    pt_selection_str = "nugt2_egtpt3_"
    e_min  = 0.3
    e_max  = None
    nu_min = 2
else:
    pt_selection_str = ''
    e_min = None
    e_max = None
    nu_min = None

maxz = 0.8

standard_stk_file = "standard_stackfile.ini"
standard_pk_file  = "standard_pkfile.ini"

if mode == 'ACTxDES':
    object_path = "/mnt/raid-cita/mlokken/data/cluster_cats/small_region_y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_lgt5_vl50_catalog.fit"
    ymask       = "/mnt/raid-cita/mlokken/data/act_ymaps/tilec_mask_4096_hpx.fits"
    map_path    = "/mnt/scratch-lustre/mlokken/sim_ymaps/"

for s in smth_str:
    print(s)
    spks =0
    for i in range(1032,2632,200):
        dlow = i
        dhi  = i+200
        theta = []
        phi   = []
        for r in range(0,nreg):
            reg_path = "/mnt/scratch-lustre/mlokken/stacking/ACT+Planck/{:d}".format(r)
            pkfile   = reg_path + "/ACTxDES_lambdagt10_reg{0}_{1}_{2}Mpc_{3}{4}_pks.fits".format(r,dlow,dhi,pt_selection_str,s)
            with fits.open(pkfile) as p:
                theta.extend(p[0].data[:,1])
                phi.extend(p[0].data[:,2])
        thetaphi = np.zeros((len(theta),2))
        thetaphi[:,0] = theta
        thetaphi[:,1] = phi
        print(len(theta))
        spks+=len(theta)
        tp_regfile = os.path.join(map_path, "thetaphi_lambdagt10_{:d}_{:d}Mpc_{:s}{:s}.txt".format(dlow,dhi,pt_selection_str,s))
        np.savetxt(tp_regfile, thetaphi)
        sim_map = map_path + "sim010_noy_hp.fits" # random map, doesn't matter
        inifile_root = "ACTxDES_lambdagt10_{:d}_{:d}Mpc_{:s}{:s}".format(dlow,dhi,pt_selection_str,s)
        pk_ini = ef.make_pk_ini_file_norot(sim_map, standard_pk_file, map_path, inifile_root, thetaphi_file=tp_regfile)
        print("Running GetPeaks on {:s}".format(pk_ini))
        subprocess.run(args=["/home/mlokken/software/COOP/mapio/GetPeaks",pk_ini])
    print(spks)
