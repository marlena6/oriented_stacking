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


h = (cosmo.H0/100.).value
# mode is which data set
# mode  = 'Buzzard'
mode = 'ACTxDES'
# mode = 'Websky'
# mode = 'GRF'

style = 'multireg'

# bmode is if it's Buzzard, is it 200 Mpc slices ('basic') or 3 z bins for GRF comparison 
Bmode = 'basic'
#Bmode = 'GRF_comp'

if mode == 'ACTxDES':
    nreg  = 48
elif mode == 'Buzzard' and Bmode == 'basic':
    nreg  = 48
elif mode == 'Websky':
    nreg  = 48
elif mode == 'GRF' or (mode == 'Buzzard' and Bmode == 'GRF_comp'):
    nreg  = 24 # actually nmaps
    zbins =[[0.15, 0.35], [0.35, 0.5], [0.5, 0.65]]
    dbins = [[cosmo.comoving_distance(z[0]).value,cosmo.comoving_distance(z[1]).value] for z in zbins]
nu_e_cuts = False
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

#width    = 500
smth     = 20 #Mpc
smth_str = ("{:.1f}".format(smth)).replace('.','pt')
#minz     = 0.1
zbins = [[0.20, 0.40], [0.4,0.55], [0.55,0.70], [0.70,0.85], [0.85,0.95]] # highest zbin unnecessary as there are no redmapper clusters there.
stack_galaxies = False # use this flag for if you want to stack the galaxy number density map as well as the y map
if stack_galaxies:
    pct = 75 # only use 75 percent of the maglim data for orientation, as the other 25 percent is stacked
else:
    pct = 100
    
stack_kappa    = True
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
    cut = 'lambda'
    cutmin   = 20
    cutminstr = 'gt{:d}'.format(cutmin)
    cutmaxstr = ''
cutstr = '{:s}'.format(cut)+cutminstr+cutmaxstr
if mode == 'Websky' and cut == 'mass':
    #maxz = z_at_value(cosmo.comoving_distance, 1432.5*u.Mpc)
    maxz = 0.3
else:
    maxz = 0.8

standard_stk_file = "standard_stackfile.ini"
standard_pk_file  = "standard_pkfile.ini"

if mode == 'ACTxDES':
    object_path = "/mnt/raid-cita/mlokken/data/cluster_cats/combined_actdes_mask_pt8_y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_lgt5_vl50_catalog.fit"
    pkmap_path  = "/mnt/raid-cita/mlokken/data/number_density_maps/maglim/"
#"/mnt/raid-cita/mlokken/data/number_density_maps/200_cmpc_slices/redmagic/overdensity/" # redmagic
    ymap        = "/mnt/raid-cita/mlokken/data/act_ymaps/ilc_SZ_yy_4096_hpx.fits"
    kappamap    = "/mnt/raid-cita/mlokken/data/des_general/kappa_bin4.fits"
    pkmask      = "/mnt/raid-cita/mlokken/data/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_hpx_4096.fits"
    ymask       = "/mnt/raid-cita/mlokken/data/act_ymaps/wide_mask_GAL080_apod_3.00_deg_4096_hpx.fits"
    outpath     = "/mnt/scratch-lustre/mlokken/stacking/ACTxDES_paper2/"

if mode == 'Buzzard':
    object_path = "/mnt/raid-cita/mlokken/buzzard/catalogs/small_region_buzzard_1.9.9_3y3a_rsshift_run_redmapper_v0.5.1_lgt05_vl50_catalog.fit"
    #pkmap_path  = "/mnt/raid-cita/mlokken/buzzard/number_density_maps/des_reg/" # redmagic
    pkmap_path  = "/mnt/raid-cita/mlokken/buzzard/number_density_maps/maglim/"
    ymap        = "/mnt/raid-cita/mlokken/buzzard/ymaps/ymap_buzzard_COMBINED_NM50_Nz16_nside4096_v01_actbeam.fits"
    pkmask      = "/mnt/raid-cita/mlokken/data/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_hpx_4096.fits"
    #y3_gold_2.2.1_wide_sofcol_run_redmapper_v6.4.22_redmagic_highdens_0.5_vlim_zmask_hpx.fit"
    ymask       = "/mnt/raid-cita/mlokken/buzzard/ymaps/my_buzzardy_mask.fits"
    outpath = "/mnt/scratch-lustre/mlokken/stacking/Buzzard_paper2/"
    
if mode == "Websky":
    object_path = "/mnt/scratch-lustre/mlokken/pkpatch/halos_fullsky_M_gt_1E13.npy"
    #pkmap_path  = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/1pt5E12_to_1E15_msun/"
    pkmap_path  = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/orient_tests/"
    ymap        = "/mnt/scratch-lustre/mlokken/pkpatch/tsz_act_beam_conv.fits"
    pkmask      = None # fullsky
    ymask       = None # fullsky
    #outpath     = "/mnt/scratch-lustre/mlokken/stacking/PeakPatch_tSZ/orient_by_1pt5E12_to_1E15_msun_halos"
    outpath     = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/orient_tests/"

if mode == "GRF":
    object_path = "/mnt/scratch-lustre/mlokken/stacking/GRF_buzzspec/"
    pk_y_path   = "/mnt/raid-cita/mlokken/GRF_buzzspec/"
    pkmask      = None
    ymask       = None
    outpath     = "/mnt/scratch-lustre/mlokken/stacking/GRF_buzzspec/"
    if nu_e_cuts:
        lrg = '_lrg'
    else:
        lrg = ''

if mode == 'Websky' and cut == 'mass':
    ra,dec,z,mass = csf.get_radecz(object_path, min_mass=cutmin, max_mass=cutmax, return_mass=True)
elif mode in ('Buzzard','ACTxDES'):
    ra,dec,z,richness = csf.get_radeczlambda(object_path)
    # limit with richness 
    print("Cutting by richness.")
    rich_cond = richness > cutmin
    ra,dec,z  = ra[rich_cond], dec[rich_cond], z[rich_cond]

if mode != "GRF":
    # load region labels
    labels = np.loadtxt("./labels_{:d}_regions_{:s}_{:s}.txt".format(nreg,mode,cutstr))
if style == 'onereg':
    # loop through regions and save all files of (theta,phi) position lists for stacking                                                                                                     
    for r in range(nreg):
        reg_path = os.path.join(outpath,str(r))
        if not os.path.exists(reg_path):
            os.mkdir(reg_path)
        in_reg = labels == r
        if mode == 'Buzzard' and Bmode == 'GRF_comp':
            thetaphi_list,dlist = csf.radec_to_thetaphi_sliced(ra[in_reg], dec[in_reg], z[in_reg], zbins[0][0], zbins[2][1], dbins[0][1]-dbins[0][0], zbins=zbins)
            print(dlist)
        else:
            thetaphi_list,dlist = csf.radec_to_thetaphi_sliced(ra[in_reg], dec[in_reg], z[in_reg], zbins=zbins)
        for i in range(len(thetaphi_list)):
            dlow, dhi = dlist[i][0], dlist[i][1]
            zlow, zhi = zbins[i][0], zbins[i][1]
            zstr_low  = str(zlow).replace(".","pt")
            zstr_hi   = str(zhi).replace(".","pt")
            tp_regfile = os.path.join(reg_path, "thetaphi_reg{:d}_z_{:s}_{:s}_{:s}.txt".format(r, zstr_low, zstr_hi, cutstr))
            np.savetxt(tp_regfile, thetaphi_list[i])

    print("Running first region in regions list.\n")

    reg = 0
    reg_path = os.path.join(outpath,str(reg))
    if not os.path.exists(reg_path):
        os.mkdir(reg_path)
    in_reg = labels == reg

    for i in range(len(thetaphi_list)):
        print("bin {:d}".format(i+1))
        dlow, dhi = dlist[i][0], dlist[i][1]
        zlow, zhi = zbins[i][0], zbins[i][1]
        zstr_low  = str(zlow).replace(".","pt")
        zstr_hi   = str(zhi).replace(".","pt")

        print("{:d} to {:d} Mpc\n".format(dlow,dhi))
        if mode in ("ACTxDES", "Websky") or (mode == "Buzzard" and Bmode == "basic"):
            pkmap = os.path.join(pkmap_path, "odmap{:s}_z_{:s}_{:s}.fits".format(pct, zstr_low, zstr_hi))
            tp_regfile = os.path.join(reg_path, "thetaphi_reg{:d}_z_{:s}_{:s}_{:s}.txt".format(reg, zstr_low, zstr_hi, cutstr))
            # make the ini files
            inifile_root = "{:s}_{:s}_reg{:d}_z_{:s}_{:s}_{:s}{:s}_{:d}pctmaglim".format(mode, cutstr, reg, zstr_low, zstr_hi, pt_selection_str, smth_str, pct)
            if stack_galaxies:
                g_inifile_root = "{:s}_ndmap_{:s}_reg{:d}_z_{:s}_{:s}_{:s}{:s}_{:d}pctmaglim".format(mode, cutstr, reg, zstr_low, zstr_hi, pt_selection_str, smth_str, pct)
            if stack_kappa:
                k_inifile_root = "{:s}_kmap_{:s}_reg{:d}_z_{:s}_{:s}_{:s}{:s}_{:d}pctmaglim".format(mode, cutstr, reg, zstr_low, zstr_hi, pt_selection_str, smth_str, pct)
            if os.path.exists(os.path.join(reg_path, inifile_root+"_stk_HankelTransform_m0.txt")):
                print("WARNING: This set of conditions has already been run. Skipping.")
            else:
                pk_ini, stk_ini = ef.make_ini_files(pkmap, ymap, tp_regfile, smth, standard_pk_file, standard_stk_file, reg_path, inifile_root, [dlow,dhi], pk_mask=pkmask, stk_mask=ymask, e_min=e_min, e_max=e_max, nu_min=nu_min, rad_Mpc=40)
                print("Running GetPeaks on {:s}".format(pk_ini))
                subprocess.run(args=["/home/mlokken/software/COOP/mapio/GetPeaks",pk_ini])
                print("Running Stack on {:s}".format(stk_ini))
                subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",stk_ini])
                # remove extraneous files
                os.remove(os.path.join(reg_path, inifile_root+"_stk.txt"))
                os.remove(os.path.join(reg_path, inifile_root+"_stk.patch"))
            if stack_galaxies and (not os.path.exists(os.path.join(reg_path, g_inifile_root+"_stk_HankelTransform_m0.txt"))):
                gstk_ini = ef.make_stk_ini_file(os.path.join(reg_path,inifile_root+"_pks.fits"), pkmap_path+'ndmap_25_z_{:s}_{:s}.fits'.format(zstr_low, zstr_hi), standard_stk_file, reg_path, g_inifile_root,[dlow,dhi], stk_mask=pkmask, rad_Mpc=40)
                print("Running Stack on {:s}".format(gstk_ini))
                subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",gstk_ini])
                # remove extraneous files
                os.remove(os.path.join(reg_path, g_inifile_root+"_stk.txt"))
                os.remove(os.path.join(reg_path, g_inifile_root+"_stk.patch"))
            if stack_kappa and (not os.path.exists(os.path.join(reg_path, k_inifile_root+"_stk_HankelTransform_m0.txt"))):
                kstk_ini = ef.make_stk_ini_file(os.path.join(reg_path,inifile_root+"_pks.fits"), kappamap, standard_stk_file, reg_path, k_inifile_root, [dlow,dhi], stk_mask="/mnt/raid-cita/mlokken/data/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_hpx_1024.fits", rad_Mpc=40)
                print("Running Stack on {:s}".format(kstk_ini))
                subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",kstk_ini])
                # remove extraneous files
                os.remove(os.path.join(reg_path, k_inifile_root+"_stk.txt"))
                os.remove(os.path.join(reg_path, k_inifile_root+"_stk.patch"))

elif style=='multireg':
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print("Rank = ", rank)

    nruns_local = nreg // size
    if rank == size-1:
        extras = nreg % size
    else:
        extras = 0
    print("Processor {:d} will run regions/maps {:d} through {:d}.\n".format(rank, rank*nruns_local, rank*nruns_local+nruns_local+extras-1))

    for m in range(nruns_local+extras):
        reg = m + rank*nruns_local
        print("Processor {:d} running region {:d}".format(rank, reg))
        reg_path = os.path.join(outpath,str(reg))
        if mode in ("ACTxDES", "Websky") or (mode == "Buzzard" and Bmode == "basic"):
            dlist = csf.dlist(zbins=zbins) # set up for preset zbins                                                                                                                           
            for i in range(len(dlist)):
                dlow, dhi = dlist[i][0], dlist[i][1]
                zlow, zhi = zbins[i][0], zbins[i][1]
                zstr_low  = str(zlow).replace(".","pt")
                zstr_hi   = str(zhi).replace(".","pt")
                print("{:d} to {:d} Mpc\n".format(dlow,dhi))
                print("Redshift bin [{:f},{:f}]".format(zbins[i][0], zbins[i][1]))
                pkmap = os.path.join(pkmap_path, "odmap_{:d}_z_{:s}_{:s}.fits".format(pct,zstr_low, zstr_hi))
                tp_regfile = os.path.join(reg_path, "thetaphi_reg{:d}_z_{:s}_{:s}_{:s}.txt".format(reg, zstr_low, zstr_hi, cutstr))
                # make the ini files
                inifile_root = "{:s}_{:s}_reg{:d}_z_{:s}_{:s}_{:s}{:s}_{:d}pctmaglim".format(mode, cutstr, reg, zstr_low, zstr_hi, pt_selection_str, smth_str, pct)
                if stack_galaxies:
                    g_inifile_root = "{:s}_ndmap_{:s}_reg{:d}_z_{:s}_{:s}_{:s}{:s}_{:d}pctmaglim".format(mode, cutstr, reg, zstr_low, zstr_hi, pt_selection_str, smth_str, pct)
                if stack_kappa:
                    k_inifile_root = "{:s}_kmap_{:s}_reg{:d}_z_{:s}_{:s}_{:s}{:s}_{:d}pctmaglim".format(mode, cutstr, reg, zstr_low, zstr_hi, pt_selection_str, smth_str, pct)
                pk_ini, stk_ini = ef.make_ini_files(pkmap, ymap, tp_regfile, smth, standard_pk_file, standard_stk_file, reg_path, inifile_root, [dlow,dhi], pk_mask=pkmask, stk_mask=ymask, e_min=e_min, e_max=e_max, nu_min=nu_min, rad_Mpc=40)
                if os.path.exists(os.path.join(reg_path, inifile_root+"_pks.fits")):
                    print("GetPeaks has already been run. Moving onto Stack.")
                else:
                    print("Running GetPeaks on {:s}".format(pk_ini))
                    subprocess.run(args=["/home/mlokken/software/COOP/mapio/GetPeaks",pk_ini])
                if os.path.exists(os.path.join(reg_path, inifile_root+"_stk_HankelTransform_m0.txt")):
                    print("WARNING: This set of conditions has already been run. Skipping.")
                else:
                    print("Running Stack on {:s}".format(stk_ini))
                    subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",stk_ini])
                    # remove extraneous files
                    os.remove(os.path.join(reg_path, inifile_root+"_stk.txt"))
                    os.remove(os.path.join(reg_path, inifile_root+"_stk.patch"))
                if stack_galaxies and (not os.path.exists(os.path.join(reg_path, g_inifile_root+"_stk_HankelTransform_m0.txt"))):
                    gstk_ini = ef.make_stk_ini_file(os.path.join(reg_path,inifile_root+"_pks.fits"), pkmap_path+'ndmap_25_z_{:s}_{:s}.fits'.format(zstr_low, zstr_hi), standard_stk_file, reg_path, g_inifile_root, [dlow,dhi], stk_mask=pkmask, rad_Mpc=40)
                    print("Running Stack on {:s}".format(gstk_ini))
                    subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",gstk_ini])
                    # remove extraneous files                    
                    os.remove(os.path.join(reg_path, g_inifile_root+"_stk.txt"))
                    os.remove(os.path.join(reg_path, g_inifile_root+"_stk.patch"))
                if stack_kappa and (not os.path.exists(os.path.join(reg_path, k_inifile_root+"_stk_HankelTransform_m0.txt"))):
                    kstk_ini = ef.make_stk_ini_file(os.path.join(reg_path,inifile_root+"_pks.fits"), kappamap, standard_stk_file, reg_path, k_inifile_root, [dlow,dhi], stk_mask="/mnt/raid-cita/mlokken/data\
/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_hpx_1024.fits", rad_Mpc=40)
                    print("Running Stack on {:s}".format(kstk_ini))
                    subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",kstk_ini])
                    # remove extraneous files
                    os.remove(os.path.join(reg_path, k_inifile_root+"_stk.txt"))
                    os.remove(os.path.join(reg_path, k_inifile_root+"_stk.patch"))

    
''' the rest is for GRf stuff, probably will be changed            
    else:
        c = 0
        for zbin in zbins:
            dlow, dhi = int(dbins[c][0]), int(dbins[c][1])
            print("{:d} to {:d} Mpc\n".format(dlow, dhi))
            if mode == 'GRF':
                if c==0:
                    smth_pk = 1.8
                    nu_min_pk=2.75
                if c==1:
                    smth_pk = 1.6
                    nu_min_pk=2.55
                if c==2:
                    smth_pk = 1.4
                    nu_min_pk=2.55
                nustr_pk = ("{:.1f}".format(nu_min_pk)).replace('.','pt')
                smth_str_pk = ("{:.1f}".format(smth_pk)).replace('.','pt')
                pkmap  = os.path.join(pk_y_path, "grf_gfield_buzzardspec_zbin{:d}_{:d}.fits".format(c+1,reg))
                ymap   = os.path.join(pk_y_path, "grf_yfield_buzzardspec_zbin{:d}_{:d}.fits".format(c+1,reg))
                tp_regfile = os.path.join(reg_path, "thetaphi_{:s}_nugt{:s}_reg{:d}_{:d}_{:d}Mpc_{:s}.txt".format(smth_str_pk,nustr_pk,reg,dlow,dhi,cutstr))
            else:
                tp_regfile = os.path.join(reg_path, "thetaphi_reg{:d}_{:d}_{:d}Mpc_{:s}.txt".format(reg, dlow, dhi, cutstr))
                np.savetxt(tp_regfile, thetaphi_list[c])
                pkmap = os.path.join(pkmap_path, "3_zbins_grf_comparison", "buzzard_redmagic_highdens_zbin{:d}_od_map_nosmooth.fits".format(c+1))
            inifile_root = "{:s}_{:s}_reg{:d}_{:d}_{:d}Mpc_{:s}{:s}_gfield".format(mode, cutstr, reg, dlow, dhi, pt_selection_str, smth_str)
            if os.path.exists(os.path.join(reg_path, inifile_root+"_stk.fits")):
                print("WARNING: This set of conditions has already been run. Skipping.")
            else:
                pk_ini, stk_ini = ef.make_ini_files(pkmap, ymap,  tp_regfile, smth, standard_pk_file, standard_stk_file, reg_path, inifile_root, [dlow,dhi], pk_mask=pkmask, stk_mask=ymask, e_min=e_min, e_max=e_max, nu_min=nu_min)
                print("Running GetPeaks on {:s}".format(pk_ini))
                subprocess.run(args=["/home/mlokken/software/COOP/mapio/GetPeaks",pk_ini])
                print("Running Stack on {:s}".format(stk_ini))
                subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",stk_ini])
                # remove extraneous files
                os.remove(os.path.join(reg_path, inifile_root+"_stk.txt"))
                os.remove(os.path.join(reg_path, inifile_root+"_stk.patch"))
            c += 1
'''
