import numpy as np
import error_analysis_funcs as ef
import os
from astropy.cosmology import Planck15 as cosmo, z_at_value
from mpi4py import MPI
import subprocess
import coop_post_processing as cpp
import astropy.units as u
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)
nmaps = 140

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print("Rank = ", rank)

h = (cosmo.H0/100.).value
mode  = 'ACTxDES'
nslices = 8 # number of slices to be combined per stack
width = 200
dmin  = 1032
dmax  = 2632

dbins_list = [(d, d + width*nslices) for d in range(dmin, dmax, width*nslices)]
smth_str  = '13pt9'
mmax = 5
nbins = 5
nu_e_cuts = False
if nu_e_cuts:
    pt_selection_str = "nugt2_egtpt3_"
else:
    pt_selection_str = ''

outpath = "/mnt/scratch-lustre/mlokken/sim_ymaps/"
nruns_local = len(dbins_list)//size
if rank == size-1:
    extras = len(dbins_list) % size
else:
    extras = 0

for d in range(nruns_local+extras):
    dbins_local = dbins_list[d+rank*nruns_local]
    dmin_local  = dbins_local[0]
    dmax_local  = dbins_local[1]
    print(dmin_local, dmax_local)
    # now that all outputs are created, combine and stack all slices for each map

    combined_stacks = []
    combined_profs  = []
    npks_allmap = []
    save_prof = {}
    
    nmaps_actual = 0
    for map in range(nmaps):
        stack_list = []
        npks_map_list = []
        skip_list = []
        counter = 0
        npks_map = 0
        hankel_arrays = []
        print("Map",map)
        hankel_count = 0
        for i in range(dmin_local, dmax_local, width):
            dlow = i
            dhi  = dlow + width
            stk_file_root = "{:s}_lambdagt10_{:d}_{:d}Mpc_{:s}{:s}_map{:d}".format(mode, dlow, dhi, pt_selection_str, smth_str,map)
            pk_file_root = "{:s}_lambdagt10_{:d}_{:d}Mpc_{:s}{:s}".format(mode, dlow, dhi, pt_selection_str, smth_str,map)
            stackfile = os.path.join(outpath,"{:s}_stk.fits".format(stk_file_root))
            pkfile    = os.path.join(outpath,"{:s}_pks.fits".format(pk_file_root))
            # count # of hankel files
            hankelcount = 0
            hankelfile0 = os.path.join(outpath,"{:s}_stk_HankelTransform_m{:d}.txt".format(stk_file_root, 0))
            if os.path.exists(hankelfile0):
                hdr, img, npks = cpp.get_img(stackfile, pkfile)
                stack_list.append(img)
                npks_map += npks
                npks_map_list.append(npks)
                hankel_ms = []
                for m in range(mmax):
                    hankelfile = os.path.join(outpath,"{:s}_stk_HankelTransform_m{:d}.txt".format(stk_file_root, m))
                    hankel = np.loadtxt(hankelfile)
                    r, Cr, Sr = hankel[:,0], hankel[:,1], hankel[:,2]
                    hankel_ms.append(Cr)
                hankel_ms = np.asarray(hankel_ms).T
                hankel_arrays.append(hankel_ms)
                hankel_count +=1
            else:
                print("{:s} doesn't exist.".format(stackfile))
                stack_list.append(0)
                npks_map_list.append(0)
                skip_list.append(counter)
                hankel_arrays.append(0)
            counter += 1
        
        if len(skip_list) == 0:
            skip_list = None
        if hankel_count == 8:
            comb_prof, rad_in_Mpc = cpp.hankel_multi(hankel_arrays, npks_map_list, dmin_local, width, 0.5*u.arcmin,  int((dmax_local-dmin_local)/width), skip_list=skip_list)
            comb_stack, rad_in_Mpc = cpp.stack_multi(stack_list, npks_map_list, dmin_local, width, 0.5*u.arcmin, int((dmax_local-dmin_local)/width), skip_list = skip_list)
            print("Number of peaks in map {:d}: {:d}.\n".format(map,npks_map))
            npks_allmap.append(npks_map)
            binsize = (rad_in_Mpc / nbins).value
            y0, binned_r0 = cpp.bin_profile(comb_prof[:,0], rad_in_Mpc, binsize)
            all_ms = np.zeros((len(binned_r0),mmax))
            all_ms[:,0] = y0
            binsize = rad_in_Mpc / len(binned_r0)
            print("binning with a size of {}".format(binsize))
            for m in range(1,mmax):
                y, binned_r = cpp.bin_profile(comb_prof[:,m], rad_in_Mpc, binsize)
                all_ms[:,m] = y
            save_prof['binnedprof_map{:d}'.format(nmaps_actual)] = all_ms
            save_prof['npks_map{:d}'.format(nmaps_actual)] = npks_map
            save_prof['profs_map{:d}'.format(nmaps_actual)] = comb_prof
            combined_stacks.append(comb_stack)
            combined_profs.append(comb_prof)
            nmaps_actual+=1
          
    print("Nmaps actual: ", nmaps_actual)
    savestr = "{:s}_lambdagt10_{:d}_{:d}Mpc_{:s}Mpc_{:s}smth".format(mode, dmin_local, dmax_local, smth_str, pt_selection_str)
    all_prof  = np.zeros(combined_profs[0].shape)
    for map in range(nmaps_actual):
        all_prof  += combined_profs[map] * npks_allmap[map]
    all_prof /= sum(npks_allmap)
    y0, binned_r0 = cpp.bin_profile(all_prof[:,0], rad_in_Mpc, binsize)
    all_ms = np.zeros((len(binned_r0), mmax))
    all_ms[:,0] = y0
    for m in range(1,mmax):
        y, binned_r = cpp.bin_profile(all_prof[:,m], rad_in_Mpc, binsize)
        all_ms[:,m] = y
    save_prof['binnedprof_allmap'] = all_ms
    save_prof['binned_r'] = binned_r
    save_prof['prof_allmap'] = all_prof
    save_prof['npks_total'] = sum(npks_allmap)
    save_file = open(os.path.join(outpath, "{:s}_m0to{:d}_profiles_{:d}maps.pkl".format(savestr, mmax, nmaps_actual)), "wb")
    pickle.dump(save_prof, save_file)
    save_file.close()
