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
smth     = [13.9]
smth_str = [("{:.1f}".format(s)).replace('.','pt') for s in smth]
print(smth_str)
minz     = z_at_value(cosmo.comoving_distance, 1032*u.Mpc)

nu_e_cuts  = True
cib_deproj = True
if cib_deproj:
    cibstr = '_cibdeproj'
else:
    cibstr = ''
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

nruns_local = 126 // size
if rank == size-1:
    extras = 128 % size
else:
    extras = 0


if mode == 'ACTxDES':
    object_path = "/mnt/raid-cita/mlokken/data/cluster_cats/small_region_y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_lgt5_vl50_catalog.fit"
    ymask       = "/mnt/raid-cita/mlokken/data/act_ymaps/tilec_mask_4096_hpx.fits"
    map_path    = "/mnt/scratch-lustre/mlokken/sim_ymaps/"

print("Processor {:d} will run maps {:d} through {:d}.\n".format(rank, rank*nruns_local, rank*nruns_local+nruns_local+extras-1))

for s in smth_str:
    for m in range(nruns_local+extras):
        simnum   = m + rank*nruns_local
        if simnum < 10:
            simfile_id = "00%d" %simnum
        elif 9.9 < simnum < 100:
            simfile_id = "0%d" %simnum
        else:
            simfile_id = str(simnum)
        sim_map = map_path + "hpx_files_noy%s/"%cibstr + "sim%s%s_noy_hp.fits" %(simfile_id, cibstr)
        print("reading sim map %s"%sim_map)
        for i in range(1032,2632,200):
            dlow = i
            dhi  = i+200
            if os.path.exists(sim_map):
                pkfile = map_path+"ACTxDES_lambdagt10_{:d}_{:d}Mpc_{:s}{:s}_pks.fits".format(dlow,dhi,pt_selection_str,s)
                print("Processor {:d} running map {:s}".format(rank, sim_map))
                inifile_root = "ACTxDES_lambdagt10_{:d}_{:d}Mpc_{:s}{:s}_map{:d}{:s}".format(dlow,dhi,pt_selection_str,s,simnum,cibstr)
                if os.path.exists(os.path.join(map_path, "sim_stacks_d56_joint_v1.2.0%s/"%cibstr, inifile_root+"_stk_HankelTransform_m0.txt")):
                    print("WARNING: This set of conditions has already been run. Skipping.")
                    pass
                else:
                    stk_ini = ef.make_stk_ini_file(pkfile, sim_map, standard_stk_file, map_path+"sim_stacks_d56_joint_v1.2.0%s/"%cibstr, inifile_root)
                    print("Running Stack on {:s}".format(stk_ini))
                    subprocess.run(args=["/home/mlokken/software/COOP_new/mapio/Stack",stk_ini])
