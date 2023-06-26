import numpy as np
import error_analysis_funcs as ef
import os
from astropy.cosmology import Planck18 as cosmo, z_at_value
import astropy.units as u
from astropy.table import Table
import subprocess
import coop_post_processing as cpp
import coop_setup_funcs as csf
from astropy.io import fits
import matplotlib.pyplot as plt
from mpi4py import MPI
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)

start = time.time()
# get the MPI ingredients
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

h = (cosmo.H0/100.).value

# ALL CHOICES COME BELOW
########################################################
# mode is which data set, uncomment one of the following
# mode  = 'Buzzard'
mode = 'ACTxDES'
# mode = 'Websky'
# mode = 'GRF'



errors = True # if true, split regions to get error estimates
nu_e_cuts = True
# Input here which maps to stack
stack_y        = True
stack_galaxies = True
if mode=='ACTxDES':
    stack_kappa = True
else:
    stack_kappa    = False # don't have any mock kappa maps
stack_mask = True # stack the mask itself to test for orientation bias
# Smooth the maps by a Gaussian with this beam FWHM
smth     = 20 #Mpc
# use overlapping bins that half-offset from each other
overlap = True
# split if you want to only use some of the galaxy data to orient and other to stack
split = False
# Split by redshift bins (True) or by bins of constant comoving distance (False)?
zsplit = False
########################################################

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
    zbins = [[0.20, 0.40], [0.4,0.55], [0.55,0.70], [0.70,0.85], [0.85,0.95]] # Maglim bins from DES Y3: highest zbin unnecessary as there are no redmapper clusters there.
else:
    width    = 200
    minz     = 0.2 # if you want to only run from the lower d limit of paper 1, input z_at_value(cosmo.comoving_distance, 1032.5*u.Mpc)
    maxz     = 1.0

smth_str = ("{:.1f}".format(smth)).replace('.','pt')

if split:
    pct = 75 # only use 75 percent of the galaxy data for orientation, as the other 25 percent is stacked
else:
    pct = 100

# cut the clusters or halos that will be stacked by richness or mass
if mode=='Websky':
    cut = 'mass'
    cutmin = 1*10**13
    # cutmax = 5*10**13
    cutmax = None
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
    #cutmin   = 10 # old way
    cutmin   = 20 # new way
    cutminstr = 'gt{:d}'.format(cutmin)
    cutmaxstr = ''
cutstr = '{:s}'.format(cut)+cutminstr+cutmaxstr

standard_stk_file = "/home/mlokken/oriented_stacking/general_code/standard_stackfile.ini"
standard_stk_file_errs = "/home/mlokken/oriented_stacking/general_code/standard_stackfile_errors.ini"
standard_pk_file  = "/home/mlokken/oriented_stacking/general_code/standard_pkfile.ini"

if mode == 'ACTxDES':
    object_path = "/mnt/raid-cita/mlokken/data/cluster_cats/redmapper2.2.1_lgt20vl50_mask_actshr1deg_des_cutpt8.fit"
    #ymap        = "/mnt/raid-cita/mlokken/data/act_ymaps/ilc_SZ_yy_4096_hpx.fits" # nothing deprojected
    pkmap_path  = "/mnt/raid-cita/mlokken/data/number_density_maps/maglim/"
    ymap        = "/mnt/raid-cita/mlokken/data/act_ymaps/ilc_SZ_deproj_cib_yy_4096_hpx.fits" # CIB deprojected
    kappamap    = "/mnt/raid-cita/mlokken/data/des_general/kappa_bin4.fits"
    gmask       = "/mnt/raid-cita/mlokken/data/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_hpx_4096.fits"
    kappamask   = "/mnt/raid-cita/mlokken/data/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_hpx_1024.fits"
    ymask       = "/mnt/raid-cita/mlokken/data/masks/outputMask_wide_mask_GAL070_apod_1.50_deg_wExtended_4096_hpx.fits"
    outpath     = "/mnt/scratch-lustre/mlokken/stacking/ACTxDES_paper2/"
    orient_mode = "maglim"
    #outpath     = "/mnt/scratch-lustre/mlokken/stacking/D56_updated_redmagic_stacks/"
    gmode       = "DES"
if mode == 'Buzzard':
    object_path = "/mnt/raid-cita/mlokken/buzzard/catalogs/combined_actdes_mask_pt8_buzzard_1.9.9_3y3a_rsshift_run_redmapper_v0.5.1_lgt05_vl50_catalog.fit"
    #pkmap_path  = "/mnt/raid-cita/mlokken/buzzard/number_density_maps/200_des_reg/" # redmagic
    pkmap_path  = "/mnt/raid-cita/mlokken/buzzard/number_density_maps/maglim/"
    ymap        = "/mnt/raid-cita/mlokken/buzzard/ymaps/ymap_buzzard_fid_hpx.fits"
    gmask       = "/mnt/raid-cita/mlokken/data/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_hpx_4096.fits"
    ymask       = "/mnt/raid-cita/mlokken/buzzard/ymaps/my_buzzardy_mask.fits"
    outpath = "/mnt/scratch-lustre/mlokken/stacking/Buzzard_paper2/"
    orient_mode = "maglim"
    gmode       = "Buzzard"
if mode == "Websky":
    object_path = "/mnt/scratch-lustre/mlokken/pkpatch/halos_fullsky_M_gt_1E13.npy"
    #pkmap_path  = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/1pt5E12_to_1E15_msun/"
    pkmap_path  = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/orient_tests/"
    ymap        = "/mnt/scratch-lustre/mlokken/pkpatch/tsz_act_beam_conv.fits"
    pkmask      = None # fullsky
    ymask       = None # fullsky
    #outpath     = "/mnt/scratch-lustre/mlokken/stacking/PeakPatch_tSZ/orient_by_1pt5E12_to_1E15_msun_halos"
    outpath     = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/orient_tests/"
ymode       = os.path.split(ymap)[1][:-5]

if errors:
    if nu_e_cuts:
        nreg  = 24
    else:
        nreg  = 48
    # need some info about the clusters
    if mode == 'Websky' and cut == 'mass':
        ra_cl,dec_cl,z_cl,mass_cl = csf.get_radecz(object_path, min_mass=cutmin, max_mass=cutmax, return_mass=True)
    elif mode in ('Buzzard','ACTxDES'):
        ra_cl,dec_cl,z_cl,richness = csf.get_radeczlambda(object_path)
        # limit with richness
        rich_cond = richness > cutmin
        ra_cl,dec_cl,z_cl  = ra_cl[rich_cond], dec_cl[rich_cond], z_cl[rich_cond]

stkpath = outpath + "orient_by_{:s}_{:d}/stacks".format(orient_mode, pct)
if not os.path.exists(stkpath):
    os.mkdir(stkpath)
dlist     = csf.dlist(minz=minz, maxz=maxz, slice_width=200)
if overlap:
    dlist_off = csf.dlist(minz=minz, maxz=maxz, slice_width=200, offset=100)
    dlist_tot = dlist+dlist_off
else:
    dlist_tot = dlist
nruns_local = len(dlist_tot) // size
if rank == size-1:
    extras = len(dlist_tot) % size
else:
    extras = 0
times = []
for n in range(nruns_local):
    i = rank*nruns_local+n
    print("Rank {:d}, bin {:d}".format(rank, i+1))
    dlow, dhi = dlist_tot[i][0], dlist_tot[i][1]
    bincent  = (dlow+dhi)/2.
    cl_dlow  = int(bincent-50)
    cl_dhi   = int(bincent+50)
    slice_included=False # set this to false, it will change to true if this slice is encompassed by a galaxy number density map
    binstr_cl   = "{:d}_{:d}Mpc".format(cl_dlow, cl_dhi)
    if zsplit:
        zlow, zhi = zbins[i][0], zbins[i][1]
        zstr_low  = str(zlow).replace(".","pt")
        zstr_hi   = str(zhi).replace(".","pt")
        binstr_orient    = "z_{:s}_{:s}".format(zstr_low, zstr_hi)
    else:
        binstr_orient = "{:d}_{:d}Mpc".format(dlow, dhi)
    print("NOW PROCESSING REDSHIFT BIN", binstr_orient)
    inifile_root = "redmapper_{:s}_{:s}_{:s}{:s}_orientXYUP_{:d}pct_{:s}_{:s}".format(cutstr, binstr_cl, pt_selection_str, smth_str, pct, orient_mode, binstr_orient)
    pksfile = os.path.join(outpath+"orient_by_{:s}_{:d}/".format(orient_mode, pct), inifile_root+"_pks.fits")
    if errors:
        labels       = np.loadtxt("/home/mlokken/oriented_stacking/general_code/labels_{:d}_regions_{:s}_{:s}.txt".format(nreg,mode,cutstr))
        # cl_zlow      = z_at_value(cosmo.comoving_distance, cl_dlow*u.Mpc).value
        # cl_zhi       = z_at_value(cosmo.comoving_distance, cl_dhi*u.Mpc).value
        # cl_inbin     = (cl_zlow<z_cl)&(z_cl<cl_zhi) # just need this for the boolean array to subselect from labels list
        # labels_inbin = labels[cl_inbin]
        #cm = plt.get_cmap('gist_rainbow')\
        for reg in range(nreg):
            start = time.time()
            regpath = os.path.join(stkpath,"{:d}".format(reg))
            if not os.path.exists(regpath):
                print("Making {:s}".format(regpath))
                os.mkdir(regpath)
            with fits.open(pksfile) as pks:
                pkdata = pks[0].data
                ncols = pkdata.shape[1]
                colnames = ["id","theta","phi","rot_angle","x_up", "y_up"]
                pd_pkdata = pd.DataFrame(data=pkdata.byteswap().newbyteorder(), columns=colnames[:ncols]) # make the peak info a dataframe so we can merge
                pd_labels = pd.DataFrame(data=labels, columns=["id","reg_label"])
                pks_w_labels = pd.merge(pd_pkdata, pd_labels, how='left', on="id")
                # use the labels to split the data
                in_reg = pks_w_labels["reg_label"] == reg
                pkdata_new = pkdata[in_reg]
                print("Peak data in region {:d} and this distance bin:".format(reg), len(pkdata_new))
                if len(pkdata_new)>0:
                    pksfile_reg = os.path.join(regpath, inifile_root+"_reg{:d}".format(reg)+"_pks.fits")
                    pks[0].data = pkdata_new
                    # save a new pks fits file with only pkdata in region, and run stack on that
                    pks.writeto(pksfile_reg, overwrite=True)
                    # for plotting
                    # angle, ra, dec = cpp.peakinfo_radec(pksfile)
                    #plt.scatter(ra[in_reg], dec[in_reg], c=cm(reg/48))
                    #if reg==47:
                    #    plt.show()
                    # make the ini files
            if len(pkdata_new)>0:
                k_inifile_root = "DES_kappa_"+inifile_root+"_reg{:d}".format(reg)
                y_inifile_root = ymode + "_"+inifile_root+"_reg{:d}".format(reg)
                m_inifile_root = "DES_mask_"+inifile_root+"_reg{:d}".format(reg)
                if stack_galaxies:
                    zbins_ndmaps = [[0.2,0.36],[0.36,0.53],[0.53,0.72],[0.72,0.94]]
                    for zbin in zbins_ndmaps:
                        if (z_at_value(cosmo.comoving_distance, cl_dlow*u.Mpc)>=zbin[0]) & (z_at_value(cosmo.comoving_distance, cl_dhi*u.Mpc)<zbin[1]):
                            zlow_str = ("{:.2f}".format(zbin[0])).replace('.', 'pt')
                            zhi_str  = ("{:.2f}".format(zbin[1])).replace('.', 'pt')
                            map_to_stack = pkmap_path+"ndmap_25_z_{:s}_{:s}.fits".format(zlow_str, zhi_str)
                            g_inifile_root = "{:s}_maglim_z_{:s}_{:s}_".format(gmode, zlow_str,zhi_str)+inifile_root+"_reg{:d}".format(reg)
                            slice_included = True
                    if slice_included:
                        if not os.path.exists(os.path.join(regpath, g_inifile_root+"_stk.fits")): # only try to do a g stack if there's a number density map that encompasses this slice 
                            gstk_ini = ef.make_stk_ini_file(pksfile_reg, map_to_stack, standard_stk_file_errs, regpath, g_inifile_root,[dlow,dhi], stk_mask=gmask, rad_Mpc=40)
                            print("Rank {:d} running Stack on {:s}".format(rank, gstk_ini))
                            subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",gstk_ini])
                            # remove extraneous files
                            os.remove(os.path.join(regpath, g_inifile_root+"_stk.txt"))
                            os.remove(os.path.join(regpath, g_inifile_root+"_stk.patch"))
                        elif os.path.exists(os.path.join(regpath, g_inifile_root+"_stk.fits")):
                            print("Galaxy map already stacked. Moving on.")
                    else:
                        print("WARNING: this slice does not fall within a number density map.")
                if stack_kappa and (not os.path.exists(os.path.join(regpath, k_inifile_root+"_stk.fits"))):
                    kstk_ini = ef.make_stk_ini_file(pksfile_reg, kappamap, standard_stk_file_errs, regpath, k_inifile_root, [dlow,dhi], stk_mask=kappamask, rad_Mpc=40)
                    print("Rank {:d} running Stack on {:s}".format(rank, kstk_ini))
                    subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",kstk_ini])
                    # remove extraneous files                                                                                                                                     
                    os.remove(os.path.join(regpath, k_inifile_root+"_stk.txt"))
                    os.remove(os.path.join(regpath, k_inifile_root+"_stk.patch"))
                elif stack_kappa and os.path.exists(os.path.join(regpath, k_inifile_root+"_stk.fits")):
                    print("Kappa map already stacked. Moving on.")
                if stack_y & (not os.path.exists(os.path.join(regpath, y_inifile_root+"_stk.fits"))):
                    stk_ini = ef.make_stk_ini_file(pksfile_reg, ymap, standard_stk_file_errs, regpath, y_inifile_root, [dlow,dhi], stk_mask=ymask, rad_Mpc=40)
                    print("Rank {:d} running Stack on {:s}".format(rank,stk_ini))
                    subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",stk_ini])
                    # remove extraneous files                                                                                                                                     
                    os.remove(os.path.join(regpath, y_inifile_root+"_stk.txt"))
                    os.remove(os.path.join(regpath, y_inifile_root+"_stk.patch"))
                elif stack_y and os.path.exists(os.path.join(regpath, y_inifile_root+"_stk.fits")):
                    print("Y map already stacked. Moving on.")
                if stack_mask and (not os.path.exists(os.path.join(regpath, m_inifile_root+"_stk.fits"))):
                    mstk_ini = ef.make_stk_ini_file(pksfile_reg, gmask, standard_stk_file_errs, regpath, m_inifile_root, [dlow,dhi], stk_mask=gmask, rad_Mpc=40)
                    print("Rank {:d} running Stack on {:s}".format(rank, mstk_ini))
                    subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",mstk_ini])
                    # remove extraneous files                                                                                                                                     
                    os.remove(os.path.join(regpath, m_inifile_root+"_stk.txt"))
                    os.remove(os.path.join(regpath, m_inifile_root+"_stk.patch"))
                elif stack_mask and os.path.exists(os.path.join(regpath, m_inifile_root+"_stk.fits")):
                    print("Mask already stacked. Moving on.")
                end = time.time()
                times.append(end-start)
    else:
        k_inifile_root = "DES_kappa_"+inifile_root
        y_inifile_root = ymode+"_"+inifile_root
        start = time.time()
        if stack_galaxies:
            zbins_ndmaps = [[0.2,0.36],[0.36,0.53],[0.53,0.72],[0.72,0.94]]
            for zbin in zbins_ndmaps:
                print(zbin, cl_dlow, cl_dhi)
                if (z_at_value(cosmo.comoving_distance, cl_dlow*u.Mpc)>zbin[0]) & (z_at_value(cosmo.comoving_distance, cl_dhi*u.Mpc)<zbin[1]):
                    zlow_str = ("{:.2f}".format(zbin[0])).replace('.', 'pt')
                    zhi_str  = ("{:.2f}".format(zbin[1])).replace('.', 'pt')
                    map_to_stack = pkmap_path+"ndmap_25_z_{:s}_{:s}.fits".format(zlow_str, zhi_str)
                    g_inifile_root = "{:s}_maglim_z_{:s}_{:s}_".format(gmode,zlow_str,zhi_str)+inifile_root
            if not os.path.exists(os.path.join(stkpath, g_inifile_root+"_stk_HankelTransform_m0.txt")):
                gstk_ini = ef.make_stk_ini_file(pksfile, map_to_stack, standard_stk_file, stkpath, g_inifile_root,[dlow,dhi], stk_mask=gmask, rad_Mpc=40)
                print("Rank {:d} running Stack on {:s}".format(rank, gstk_ini))
                subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",gstk_ini])
                # remove extraneous files                                                                                                                                     
                os.remove(os.path.join(stkpath, g_inifile_root+"_stk.txt"))
                os.remove(os.path.join(stkpath, g_inifile_root+"_stk.patch"))
        if stack_kappa and (not os.path.exists(os.path.join(stkpath, k_inifile_root+"_stk_HankelTransform_m0.txt"))):
            kstk_ini = ef.make_stk_ini_file(pksfile, kappamap, standard_stk_file, stkpath, k_inifile_root, [dlow,dhi], stk_mask=kappamask, rad_Mpc=40)
            print("Rank {:d} running Stack on {:s}".format(rank, kstk_ini))
            subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",kstk_ini])
            # remove extraneous files                                                                                                                                     
            os.remove(os.path.join(stkpath, k_inifile_root+"_stk.txt"))
            os.remove(os.path.join(stkpath, k_inifile_root+"_stk.patch"))
        if stack_y & (not os.path.exists(os.path.join(stkpath, y_inifile_root+"_stk_HankelTransform_m0.txt"))):
            stk_ini = ef.make_stk_ini_file(pksfile, ymap, standard_stk_file, stkpath, y_inifile_root, [dlow,dhi], stk_mask=ymask, rad_Mpc=40)
            print("Rank {:d} running Stack on {:s}".format(rank,stk_ini))
            subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",stk_ini])
            # remove extraneous files                                                                                                                                     
            os.remove(os.path.join(stkpath, y_inifile_root+"_stk.txt"))
            os.remove(os.path.join(stkpath, y_inifile_root+"_stk.patch"))
        end = time.time()
        times.append(end-start)
# print("Rank, time list :", rank, times)
# print("Rank, average time per loop:", rank, np.average(times))
