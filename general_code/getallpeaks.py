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
import sys

h = (cosmo.H0/100.).value
# mode is which data set
###################################
# #####ALL OPTIONS ARE SET HERE######
###################################

# mode  = 'Buzzard'
# mode  = 'Cardinal'
# mode = 'ACTxDES'
mode = 'ACTxDESI'

# mode = 'Websky'
# mode = 'GRF'

nu_e_cuts = True
# use overlapping bins that half-offset from each other
overlap = True

# Split by pre-defined redshift bins rather than by bins of constant comoving distance?
zsplit = False

# split if you want to only use some of the galaxy data to orient and other to stack
split = False

# Smooth the maps by a Gaussian with this beam FWHM
smth     = 20 #Mpc

orient = True # usually true, set to False if you want to randomly orient
xyup = True # manually update the standard_pkfile.ini to 'SYMMETRIC' if you want to make this false
##################################
##################################

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
    if mode=='GRF':
        pt_selection_str = 'allnu_alle_'

if zsplit:
    # enter the pre-defined redshift bins here
    zbins = [[0.20, 0.40], [0.4,0.55], [0.55,0.70], [0.70,0.85], [0.85,0.95]] # Maglim bins from DES Y3: highest zbin unnecessary as there are no redmapper clusters there.
else:
    width = 200
    minz = 0.4
    maxz = 1.1
    # minz     = 0.2 # if you want to only run from the lower d limit of paper 1, input z_at_value(cosmo.comoving_distance, 1032.5*u.Mpc)
    # maxz     = 1.0

smth_str = ("{:.1f}".format(smth)).replace('.','pt')

if split:
    pct = 75 # only use 75 percent of the galaxy data for orientation, as the other 25 percent is stacked
else:
    pct = 100

if overlap & zsplit:
    sys.exit("Cannot run overlapping bin mode when running explicit redshift bins.")

if xyup:
    style = "XYUP"
else:
    style = ""
    

# cut the clusters or halos that will be stacked by richness or mass
if mode=='Websky':
    cut = 'mass'
    cutmin = 1*10**13
    cutmax = 5*10**13
    # cutmax = None
    if cutmin is not None:
        cutminstr = 'gt{:.0e}'.format(cutmin)
    else:
        cutminstr = ''
    if cutmax is not None:
        cutmaxstr = 'lt{:.0e}'.format(cutmax)
    else:
        cutmaxstr = ''
else:
    # cut = 'lambda' # for DES clusters
    cut = 'nocut'
    # cutmin   = 10 # old way
    # cutmin   = 20 # new way for DES
    # cutminstr = 'gt{:d}'.format(cutmin) # for DES
    cutminstr = ''
    cutmaxstr = ''
    
cutstr = '{:s}'.format(cut)+cutminstr+cutmaxstr

if xyup:
    standard_pk_file  = "/home/mlokken/oriented_stacking/general_code/standard_pkfile.ini"
else:
    standard_pk_file  = "/home/mlokken/oriented_stacking/general_code/standard_pkfile_symmetric.ini"
if mode == 'ACTxDES':
    # object_path = "/mnt/raid-cita/mlokken/data/cluster_cats/redmapper2.2.1_lgt20vl50_mask_actshr1deg_des_cutpt8.fit"
    object_path = "/mnt/raid-cita/mlokken/data/cluster_cats/redmapper2.2.1_lgt5vl50_mask_actshr1deg_des_cutpt8.fit"
    # object_path = "/mnt/raid-cita/mlokken/data/cluster_cats/paper1/small_region_y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_lgt5_vl50_catalog.fit"
    pkmap_path  = "/mnt/raid-cita/mlokken/data/number_density_maps/maglim/" # maglim is the preferred mode
    # pkmap_path  = "/mnt/raid-cita/mlokken/data/number_density_maps/200_cmpc_slices/redmagic_updated_nov2021/" #redmagic new without xlens issues
    pkmask      = "/mnt/raid-cita/mlokken/data/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_hpx_4096.fits"
    outpath     = "/mnt/scratch-lustre/mlokken/stacking/ACTxDES_paper2/"
    # outpath     = "/mnt/scratch-lustre/mlokken/stacking/for_dhayaa/"
    orient_mode = "maglim"

if mode == 'ACTxDESI':
    # object_path = "/mnt/raid-cita/mlokken/data/cluster_cats/redmapper2.2.1_lgt20vl50_mask_actshr1deg_des_cutpt8.fit"
    object_path = "/mnt/raid-cita/mlokken/data/desi/small_region_LRG_clustering.dat.fits"
    # object_path = "/mnt/raid-cita/mlokken/data/cluster_cats/paper1/small_region_y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_lgt5_vl50_catalog.fit"
    pkmap_path  = "/mnt/raid-cita/mlokken/data/number_density_maps/desi/" # maglim is the preferred mode
    # pkmap_path  = "/mnt/raid-cita/mlokken/data/number_density_maps/200_cmpc_slices/redmagic_updated_nov2021/" #redmagic new without xlens issues
    pkmask      = "/mnt/raid-cita/mlokken/data/masks/desi_mask_bright.fits"
    outpath     = "/mnt/scratch-lustre/mlokken/stacking/ACTxDESI_LRG/"
    # outpath     = "/mnt/scratch-lustre/mlokken/stacking/for_dhayaa/"
    orient_mode = "desi_lrg"
        
if mode == 'Buzzard':
    object_path = "/mnt/raid-cita/mlokken/buzzard/catalogs/combined_actdes_mask_pt8_buzzard_1.9.9_3y3a_rsshift_run_redmapper_v0.5.1_lgt05_vl50_catalog.fit"
    orient_mode = "maglim"
    if orient_mode=="redmagic":
        pkmap_path  = "/mnt/raid-cita/mlokken/buzzard/number_density_maps/redmagic/"
    elif orient_mode=="maglim":
        pkmap_path  = "/mnt/raid-cita/mlokken/buzzard/number_density_maps/maglim/"
    elif orient_mode=="maglim_truez":
        pkmap_path  = "/mnt/raid-cita/mlokken/buzzard/number_density_maps/maglim_truez/" # maps made using the original, true redshifts
    pkmask      = "/mnt/raid-cita/mlokken/data/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_hpx_4096.fits"
    #y3_gold_2.2.1_wide_sofcol_run_redmapper_v6.4.22_redmagic_highdens_0.5_vlim_zmask_hpx.fit"
    outpath = "/mnt/scratch-lustre/mlokken/stacking/Buzzard_paper2/"
    
if mode == 'Cardinal':
    object_path = "/mnt/raid-cita/mlokken/cardinal/maglim_mask_Cardinal-3Y6a_v2.0_run_run_redmapper_v0.8.1_lgt20_vl50_catalog.fit"
    orient_mode = "maglim"
    if orient_mode=="redmagic":
        pkmap_path  = "/mnt/raid-cita/mlokken/cardinal/number_density_maps/redmagic/"
    elif orient_mode=="maglim":
        pkmap_path  = "/mnt/raid-cita/mlokken/cardinal/number_density_maps/maglim/"
    pkmask      = "/mnt/raid-cita/mlokken/cardinal/cardinal_maglim_mask.fits"
    #y3_gold_2.2.1_wide_sofcol_run_redmapper_v6.4.22_redmagic_highdens_0.5_vlim_zmask_hpx.fit"
    outpath = "/mnt/scratch-lustre/mlokken/stacking/Cardinal_paper2/"
    
if mode == "Websky":
    object_path = "/mnt/scratch-lustre/mlokken/pkpatch/halos_fullsky_M_gt_1E13.npy"
    #pkmap_path  = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/1pt5E12_to_1E15_msun/"
    pkmap_path  = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/orient_tests/"
    pkmask      = None # fullsky
    ymask       = None # fullsky
    #outpath     = "/mnt/scratch-lustre/mlokken/stacking/PeakPatch_tSZ/orient_by_1pt5E12_to_1E15_msun_halos"
    outpath     = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/orient_tests/"

if mode == 'Websky' and cut == 'mass':
    ra,dec,z,mass,id = csf.get_radecz(object_path, min_mass=cutmin, max_mass=cutmax, return_mass=True, return_id=True)
elif mode in ('Buzzard','ACTxDES','Cardinal'):
    ra,dec,z,richness, id = csf.get_radeczlambda(object_path, return_id=True)
    # limit with richness 
    print("Cutting by richness.")
    rich_cond = richness > cutmin
    ra,dec,z,id  = ra[rich_cond], dec[rich_cond], z[rich_cond], id[rich_cond]
elif mode == 'ACTxDESI':
    ra,dec,z,id = csf.get_radecz(object_path, return_id=True)
if overlap:
    peakspath = outpath + "orient_by_{:s}_{:d}".format(orient_mode, pct)
    if not os.path.exists(peakspath):
        os.mkdir(peakspath)

    theta,phi   = csf.DeclRatoThetaPhi(dec,ra)
    thetaphi    = np.zeros((len(theta),2))
    thetaphi[:,0] = theta
    thetaphi[:,1] = phi
    dlist     = csf.dlist(minz=minz, maxz=maxz, slice_width=200)
    dlist_off = csf.dlist(minz=minz, maxz=maxz, slice_width=200, offset=100)
    dlist_tot = dlist+dlist_off
    print(dlist_tot)
    for i in range(len(dlist_tot)):
        dbin = dlist_tot[i]
        dlow,dhi = dbin[0], dbin[1]
        bincent  = (dlow+dhi)/2.
        # find clusters only within 100 cMpc of the bin center
        cl_dlow  = int(bincent-50)
        cl_dhi   = int(bincent+50)
        cl_zlow  = z_at_value(cosmo.comoving_distance, cl_dlow*u.Mpc).value
        cl_zhi   = z_at_value(cosmo.comoving_distance, cl_dhi*u.Mpc).value
        print("Finding clusters within 50 cMpc of {:.0f} Mpc".format(bincent))
        print("In redshift space, this is between {:.2f} and {:.2f}.".format(cl_zlow,cl_zhi))
        binstr_cl   = "{:d}_{:d}Mpc".format(cl_dlow, cl_dhi)
        cl_inbin    = (cl_zlow<z)&(z<cl_zhi)
        thetaphi_bin=thetaphi[cl_inbin]
        id_bin = np.array([id[cl_inbin]])
        savebin = np.concatenate((id_bin.T,thetaphi_bin),axis=1)
        tp_file     = os.path.join(peakspath, "thetaphi_{:s}_{:s}.txt".format(binstr_cl, cutstr))
        np.savetxt(tp_file, savebin)
        print("Orienting by surrounding galaxies from {:d} to {:d} Mpc\n".format(dlow,dhi))
        binstr_orient = "{:d}_{:d}Mpc".format(dlow, dhi)
        pkmap = os.path.join(pkmap_path, "odmap_{:d}_{:s}.fits".format(pct, binstr_orient))
        print("reading galaxy data from {:s}".format(pkmap))
        # make the ini files
        if orient:
            orientstr="orient{:s}_{:d}pct_{:s}_{:s}".format(style, pct, orient_mode, binstr_orient)
        else:
            orientstr="randrot"
        if mode=="ACTxDES":
            inifile_root = "redmapper_{:s}_{:s}_{:s}{:s}_{:s}".format(cutstr, binstr_cl, pt_selection_str, smth_str, orientstr)
        elif mode=="ACTxDESI":
            inifile_root = "lrg_{:s}_{:s}_{:s}{:s}_{:s}".format(cutstr, binstr_cl, pt_selection_str, smth_str, orientstr)
        if not os.path.exists(os.path.join(peakspath, inifile_root+"_pks.fits")):
            pk_ini = ef.make_pk_ini_file(pkmap, smth, standard_pk_file, peakspath, inifile_root, [dlow,dhi], thetaphi_file=tp_file, pk_mask=pkmask, e_min=e_min, e_max=e_max, nu_min=nu_min)
            if not orient:
                #overwrite pk_ini with norot option
                pk_ini = ef.make_pk_ini_file(pkmap, smth, standard_pk_file, peakspath, inifile_root, [dlow,dhi], thetaphi_file=tp_file, pk_mask=pkmask, e_min=e_min, e_max=e_max, nu_min=nu_min, norot=True)
            print("Running GetPeaks on {:s}".format(pk_ini))
            subprocess.run(args=["/home/mlokken/software/COOP/mapio/GetPeaks",pk_ini])
        else:
            print("Already run. Moving on.")
else: # no bin overlap and use all clusters in each bin
    if zsplit:
        thetaphi_list,dlist = csf.radec_to_thetaphi_sliced(ra, dec, z, zbins=zbins)
    else:
        thetaphi_list,dlist = csf.radec_to_thetaphi_sliced(ra, dec, z, minz, maxz, width)

    peakspath = outpath + "orient_by_{:s}_{:d}".format(orient_mode, pct) 
    if not os.path.exists(peakspath):
        os.mkdir(peakspath)

    for i in range(len(thetaphi_list)):
        dlow, dhi = dlist[i][0], dlist[i][1]
        if zsplit:
            zlow, zhi = zbins[i][0], zbins[i][1]
            zstr_low  = str(zlow).replace(".","pt")
            zstr_hi   = str(zhi).replace(".","pt")
            binstr    = "z_{:s}_{:s}".format(zstr_low, zstr_hi)
        else:
            binstr    = "{:d}_{:d}Mpc".format(dlow,dhi)
        tp_file = os.path.join(peakspath, "thetaphi_{:s}_{:s}.txt".format(binstr, cutstr))
        np.savetxt(tp_file, thetaphi_list[i])

        print("{:d} to {:d} Mpc\n".format(dlow,dhi))
        pkmap = os.path.join(pkmap_path, "odmap_{:d}_{:s}.fits".format(pct, binstr))
        # make the ini files
        if orient:
            orientstr="orient{:s}_{:d}pct_{:s}".format(style, pct, orient_mode)
        else:
            orientstr="randrot"
        inifile_root = "{:s}_{:s}_{:s}{:s}_{:s}".format(cutstr, binstr, pt_selection_str, smth_str, orientstr)
        if not os.path.exists(os.path.join(peakspath, inifile_root+"_pks.fits")):
            pk_ini = ef.make_pk_ini_file(pkmap, smth, standard_pk_file, peakspath, inifile_root, [dlow,dhi], thetaphi_file=tp_file, pk_mask=pkmask, e_min=e_min, e_max=e_max, nu_min=nu_min)
            if not orient:
                #overwrite pk_ini with norot option
                pk_ini = ef.make_pk_ini_file(pkmap, smth, standard_pk_file, peakspath, inifile_root, [dlow,dhi], thetaphi_file=tp_file, pk_mask=pkmask, e_min=e_min, e_max=e_max, nu_min=nu_min, norot=True)
            print("Running GetPeaks on {:s}".format(pk_ini))
            subprocess.run(args=["/home/mlokken/software/COOP/mapio/GetPeaks",pk_ini])
        else:
            print("Already run. moving on.")
