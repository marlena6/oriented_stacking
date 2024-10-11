import coop_setup_funcs as csf
import coop_post_processing as cpp
import os
import healpy as hp
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# run with 12 cores
dbin_select = rank%4 # 0, 1, 2, 3
if rank<=3:
    mode = "Buzzard"
if rank>3 and rank<=7:
    mode = "DES"
if rank>7:
   mode = "Cardinal"
print("rank, mode, dbin select", rank, mode, dbin_select)

width    = 200
minz     = 0.2
maxz     = 1.0

allz_peaks_info = []


if mode == 'Buzzard':
    object_path = "/mnt/raid-cita/mlokken/buzzard/catalogs/combined_actdes_mask_pt8_buzzard_1.9.9_3y3a_rsshift_run_redmapper_v0.5.1_lgt05_vl50_catalog.fit"
    gmask       = hp.read_map("/mnt/raid-cita/mlokken/data/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_hpx_4096.fits")
    pkmap_path  = "/mnt/raid-cita/mlokken/buzzard/number_density_maps/maglim/"
    outpath     = "/mnt/scratch-lustre/mlokken/stacking/Buzzard_paper2/orient_by_maglim_75/"

elif mode == "Cardinal":
    object_path = "/mnt/raid-cita/mlokken/cardinal/maglim_mask_Cardinal-3Y6a_v2.0_run_run_redmapper_v0.8.1_lgt20_vl50_catalog.fit"
    gmask       = hp.read_map("/mnt/raid-cita/mlokken/cardinal/cardinal_maglim_mask.fits")
    pkmap_path  = "/mnt/raid-cita/mlokken/cardinal/number_density_maps/maglim/"
    outpath     = "/mnt/scratch-lustre/mlokken/stacking/Cardinal_paper2/orient_by_maglim_75/"
    
elif mode == "DES":
    object_path = "/mnt/raid-cita/mlokken/data/cluster_cats/redmapper2.2.1_lgt20vl50_mask_actshr1deg_des_cutpt8.fit"
    gmask       = hp.read_map("/mnt/raid-cita/mlokken/data/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_hpx_4096.fits")
    pkmap_path  = "/mnt/raid-cita/mlokken/data/number_density_maps/maglim/"
    outpath     = "/mnt/scratch-lustre/mlokken/stacking/ACTxDES_paper2/orient_by_maglim_75/"


dbins = [[893, 1393], [1493, 1993], [2093, 2593], [2693,3193]]

savefile = outpath+f"peaks_info_rm_lgt20_nugt2_egtpt3_20Mpc_{dbins[dbin_select][0]}_{dbins[dbin_select][1]}Mpc_{mode}.npy"
# if not os.path.exists(savefile):
print("Will save in,", outpath+f"peaks_info_rm_lgt20_nugt2_egtpt3_20Mpc_{dbins[dbin_select][0]}_{dbins[dbin_select][1]}Mpc_{mode}.npy")

for dlow in range(dbins[dbin_select][0], dbins[dbin_select][1], 100):
    d = [dlow, dlow+100]
    print("Working on ", d)
    gald = [dlow-50, dlow+150]
    for file in os.listdir(outpath):
        if str(d[0])+"_"+str(d[1]) in file and "nugt2" in file and "75pct" in file and "orientXYUP" in file and "pks" in file and "cc" in file and file.endswith(".fits"):
            print("Getting peak info for ", file)
            rot_angle, ra, dec, parityx, parityy, peakid = cpp.get_peakinfo(os.path.join(outpath, file))
    for file2 in os.listdir(pkmap_path):
        if str(gald[0])+"_"+str(gald[1]) in file2 and "odmap_75" in file2 and "cc" in file2 and file2.endswith(".fits"):
            if "AMPLITUDE" in file2:
                print("Getting nu for ", file2)
                nu_vals = csf.get_nu(os.path.join(pkmap_path,file2),ra,dec, mask=gmask)
            if "ECC" in file2:
                print("Getting ecc and x for ", file2)
                e_vals, x_vals = csf.get_x_e(os.path.join(pkmap_path,file2), ra, dec, gmask)

    peaks_info = np.zeros((len(rot_angle), 9))
    peaks_info[:,0] = rot_angle
    peaks_info[:,1] = ra
    peaks_info[:,2] = dec
    peaks_info[:,3] = e_vals
    peaks_info[:,4] = x_vals
    peaks_info[:,5] = nu_vals
    peaks_info[:,6] = parityx
    peaks_info[:,7] = parityy
    peaks_info[:,8] = peakid

    allz_peaks_info.extend(peaks_info)

    # save the peak info
print("Saving peak info")
allz_peaks_info = np.array(allz_peaks_info)
np.save(outpath+f"/peaks_info_rm_lgt20_nugt2_egtpt3_20Mpc_{dbins[dbin_select][0]}_{dbins[dbin_select][1]}Mpc_{mode}_cc.npy", allz_peaks_info)
# else:
#     print("Already exists. Terminating rank {}.".format(dbin_select))
