import numpy as np
import error_analysis_funcs as ef
import os
from astropy.cosmology import Planck18 as cosmo, z_at_value
from astropy.io import fits
import subprocess
import coop_post_processing as cpp
import coop_setup_funcs as csf
import astropy.units as u
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
import pickle
import astropy.constants as const
import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)


###############################################
########  USER INPUT PARAMETERS  ##############

# mode = 'GRF'
# mode  = 'Buzzard'
mode = 'ACTxDES'
# mode   = 'Websky'

errors = True # if true, split regions to get error estimates
mmax   = 5 # maximum multipole moment to use in the decomposition
zsplit = False # Split by redshift bins or by bins of constant comoving distance?
# Input here which maps to stack                                                                                                                                                                         
stack_y        = False
stack_galaxies = False
stack_kappa    = True

# input number of bins (should be approx sqrt(nreg)
nbins = 7 #sqrt(48<7 so should only do 6 bins)
# nbins = 14
nu_e_cuts = True # use threshold of nu>2 and e>0.3

# split if you want to only use some of the galaxy data to orient and other to stack
split   = True
# use overlapping bins that half-offset from each other 
overlap = True
# Smooth the maps by a Gaussian with this beam FWHM                                                                                                                                                       
smth     = 20 #Mpc                                                                                                                                                                                        

################################################
################################################

def consolidate_data_interactive(cl_dlist, dlist_tot, nlow, nhi, mapstr, cutstr, pt_selection_str, smth_str, pct, orient_mode, overlap=True, weights=None):
    stack_list    = []
    hankel_arrays = []
    npks_list     = []
    if overlap:
        cl_dlow_abs = cl_dlist[nlow][0]
        cl_dhi_abs  = cl_dlist[nhi][1]
        dlow_abs    = dlist_tot[nlow][0]
        dhi_abs     = dlist_tot[nhi][1]
    print("You have selected to combine stacks of clusters between {:d} and {:d} Mpc.".format(cl_dlow_abs,cl_dhi_abs))
    c = 0
    
    for dbin in dlist_tot[nlow:nhi+1]:
        dlow, dhi = dbin[0], dbin[1]
        bincent  = (dlow+dhi)/2.
        cl_dlow  = int(bincent-50)
        cl_dhi   = int(bincent+50)
        binstr_cl   = "{:d}_{:d}Mpc".format(cl_dlow, cl_dhi)
        if zsplit:
            zlow, zhi = zbins[i][0], zbins[i][1]
            zstr_low  = str(zlow).replace(".","pt")
            zstr_hi   = str(zhi).replace(".","pt")
            binstr_orient    = "z_{:s}_{:s}".format(zstr_low, zstr_hi)
        else:
            binstr_orient    = "{:d}_{:d}Mpc".format(dlow,dhi)
        print("{:d} to {:d} Mpc\n".format(dlow,dhi))
        file_root = "redmapper_{:s}_{:s}_{:s}{:s}_orientXYUP_{:d}pct_{:s}_{:s}".format(cutstr, binstr_cl, pt_selection_str, smth_str, pct, orient_mode, binstr_orient)
        pksfile = os.path.join(outpath+"orient_by_{:s}_{:d}/".format(orient_mode, pct), file_root+"_pks.fits")
        stackfile = os.path.join(stkpath,"{:s}_{:s}_stk.fits".format(mapstr,file_root))
        if os.path.exists(stackfile):
            hdr, img, npks = cpp.get_img(stackfile, pksfile)
            stack_list.append(img)
            hankel_ms = []
            for m in range(mmax):
                hankelfile = os.path.join(stkpath,"{:s}_{:s}_stk_HankelTransform_m{:d}.txt".format(mapstr, file_root, m))
                hankel = np.loadtxt(hankelfile)
                r, Cr, Sr = hankel[:,0], hankel[:,1], hankel[:,2]
                hankel_ms.append(Cr)
            hankel_ms = np.asarray(hankel_ms).T
            hankel_arrays.append(hankel_ms)
            npks_list.append(npks)
        else:
            print("{:s} doesn't exist.".format(stackfile))
        mid = (dhi + dlow)/2. * u.Mpc
        z = z_at_value(cosmo.comoving_distance, mid)
        inifile = fits.open(stackfile)
        hdr = inifile[0].header
        radius = float(hdr['RADIUS'].replace('=',''))
        res    = float(hdr['RES'].replace('=',''))
        arcmin_per_pix = (radius*u.deg/res).to(u.arcmin)
        c += 1
    
    if weights is not None:
        wgts = weights[nlow:nhi+1]
    else:
        wgts = None
    if len(stack_list)>0:
        comb_stack, rad_in_Mpc   = cpp.stack_multi_same_csize(stack_list, npks_list, cl_dlow_abs, 100, arcmin_per_pix, multiplier=wgts)
        comb_prof, rad_in_Mpc    = cpp.hankel_multi_same_csize(hankel_arrays, npks_list, cl_dlow_abs, 100, arcmin_per_pix, multiplier=wgts)
        binsize = (rad_in_Mpc / nbins).value
        print("binning with a size of {}".format(binsize))
        r = np.arange(1,len(comb_prof[:,0]))
        y0, binned_r0 = cpp.bin_profile(r, comb_prof[:,0][:-1], rad_in_Mpc, binsize)
        comb_prof_binned = np.zeros((len(binned_r0),mmax))
        comb_prof_binned[:,0] = y0
        for m in range(1,mmax):
            y, binned_r = cpp.bin_profile(r, comb_prof[:,m][:-1], rad_in_Mpc, binsize)
            comb_prof_binned[:,m] = y
        save_prof['binnedprof'] = comb_prof_binned
        save_prof['binned_r']   = binned_r
        save_prof['npks_list']  = npks_list
        save_prof['npks']       = np.sum(npks_list)
        save_prof['prof']       = comb_prof
        save_prof['r']          = [0]+r * (rad_in_Mpc.value/len(r)) # r times Mpc / pix
        save_prof['rad_in_Mpc'] = rad_in_Mpc
        file_root = "redmapper_{:s}_combined_{:d}_{:d}Mpc_{:s}{:s}_orientXYUP_{:d}pct_{:s}_{:d}_{:d}Mpc".format(cutstr, cl_dlow_abs, cl_dhi_abs, pt_selection_str, smth_str, pct, orient_mode, dlow_abs, dhi_abs)
        savestr = mapstr+"_"+file_root
        np.savetxt(os.path.join(outpath,savestr+"_stack.txt"), comb_stack)
        save_file = open(os.path.join(outpath, "{:s}_m0to{:d}_profiles.pkl".format(savestr, mmax)), "wb")
        print(savestr)
        print(outpath, savestr)
        pickle.dump(save_prof, save_file)
        save_file.close()
    else:
        print('nothing to save for this region. Moving on.')

def collect_regions(nreg, cl_dlist, dlist_tot,nlow, nhi, mapstr, cutstr, pt_selection_str, smth_str, pct, orient_mode, overlap=True, weight_regions=False, mmax=5, nbins=7, weights=None):
    # this function collects the stack information from each region, combining stacks across a given distance range
    # it may be split by region for MPI purposes
    combstack_list = []
    npks_list      = []
    Cr_list        = []
    Cr_bin_list    = []
    if overlap:
        cl_dlow_abs = cl_dlist[nlow][0]
        cl_dhi_abs  = cl_dlist[nhi][1]
        dlow_abs    = dlist_tot[nlow][0]
        dhi_abs     = dlist_tot[nhi][1]
        # dlow_abs and dhi_abs are the distance limits of the galaxy sample, while cl_dlow_abs and cl_dhi_abs are the limits of the cluster sample
        # stacks will be combined across this distance range by rescaling and stacking different thin distance bins
        
    # check if the file already exists
    file_root = "redmapper_{:s}_combined_{:d}_{:d}Mpc_{:s}{:s}_orientXYUP_{:d}pct_{:s}_{:d}_{:d}Mpc".format(cutstr, cl_dlow_abs, cl_dhi_abs, pt_selection_str, smth_str, pct, orient_mode, dlow_abs, dhi_abs)
    savestr = mapstr+"_"+file_root+"_{:d}reg_{:d}_{:d}".format(nreg, rank*nruns_local, (rank+1)*nruns_local-1)
    
    if not os.path.exists((os.path.join(outpath, "temp_reg", "{:s}_m0to{:d}_profiles.pkl".format(savestr, mmax)))):
        if weights is not None:
            wgts_list = []
        # first find the base shape, in case some regions don't have any entries in the lowest dbin
        for reg in range(nreg):
            reg_path  = os.path.join(stkpath,str(reg))
            dbin = dlist_tot[nlow]
            dlow, dhi = dbin[0], dbin[1]
            bincent  = (dlow+dhi)/2.
            cl_dlow  = int(bincent-50)
            cl_dhi   = int(bincent+50)
            binstr_cl   = "{:d}_{:d}Mpc".format(cl_dlow, cl_dhi)
            binstr_orient = "{:d}_{:d}Mpc".format(dlow, dhi)
            file_root = "redmapper_{:s}_{:s}_{:s}{:s}_orientXYUP_{:d}pct_{:s}_{:s}".format(cutstr, binstr_cl, pt_selection_str, smth_str, pct, orient_mode, binstr_orient)
            stackfile = os.path.join(reg_path,"{:s}_{:s}_reg{:d}_stk.fits".format(mapstr,file_root,reg))
            pkfile    = os.path.join(reg_path,"{:s}_reg{:d}_pks.fits".format(file_root, reg))
            if os.path.exists(stackfile):
                hdr, img, npks = cpp.get_img(stackfile, pkfile)
                base_shape = img.shape
                break
        # done finding the base shape
        # now collect the stacks from each region
        reg_list = []
        for n in range(nruns_local):
            reg = rank*nruns_local+n
            reg_path  = os.path.join(stkpath,str(reg))
            stack_list = []
            npks_reg_list = []
            skip_list = []
            counter = 0
            npks_reg = 0
            print("Region",reg)
            for dbin in dlist_tot[nlow:nhi+1]:
                dlow, dhi = dbin[0], dbin[1]
                bincent  = (dlow+dhi)/2.
                cl_dlow  = int(bincent-50)
                cl_dhi   = int(bincent+50)
                binstr_cl   = "{:d}_{:d}Mpc".format(cl_dlow, cl_dhi)
                if zsplit:
                    zlow, zhi = zbins[i][0], zbins[i][1]
                    zstr_low  = str(zlow).replace(".","pt")
                    zstr_hi   = str(zhi).replace(".","pt")
                    binstr_orient    = "z_{:s}_{:s}".format(zstr_low, zstr_hi)
                else:
                    binstr_orient    = "{:d}_{:d}Mpc".format(dlow,dhi)
                print("{:d} to {:d} Mpc\n".format(dlow,dhi))
                file_root = "redmapper_{:s}_{:s}_{:s}{:s}_orientXYUP_{:d}pct_{:s}_{:s}".format(cutstr, binstr_cl, pt_selection_str, smth_str, pct, orient_mode, binstr_orient)
                stackfile = os.path.join(reg_path,"{:s}_{:s}_reg{:d}_stk.fits".format(mapstr,file_root,reg))
                pkfile    = os.path.join(reg_path,"{:s}_reg{:d}_pks.fits".format(file_root, reg))
                if os.path.exists(stackfile):
                    hdr, img, npks = cpp.get_img(stackfile, pkfile)
                    stack_list.append(img)
                    npks_reg += npks
                    npks_reg_list.append(npks)
                    inifile = fits.open(stackfile)
                    hdr = inifile[0].header
                    radius = float(hdr['RADIUS'].replace('=',''))
                    res    = float(hdr['RES'].replace('=',''))
                    arcmin_per_pix = (radius*u.deg/res).to(u.arcmin)
                    if weights is not None:
                        wgts_list.append(weights[nlow+counter])
                else:
                    print("{:s} doesn't exist.".format(stackfile))
                    stack_list.append(0)
                    npks_reg_list.append(0)
                    skip_list.append(counter)
                    if weights is not None:
                        wgts_list.append(0)
                counter += 1
            
            if len(skip_list) == 0:
                skip_list = None
            if (skip_list is None) or (len(skip_list) != (nhi-nlow+1)): # there are peaks in this region    
                if weights is not None:
                    wgts = weights[nlow:nhi+1]
                else:
                    wgts = None
                reg_list.append(reg)
                # following operations are to combine the distance bins for this region
                comb_stack, rad_in_Mpc = cpp.stack_multi_same_csize(stack_list, npks_reg_list, cl_dlow_abs, 100, arcmin_per_pix, skip_list = skip_list, base_shape=base_shape, multiplier=wgts)
                combstack_list.append(comb_stack)
                print("shape of combined stack", comb_stack.shape)
                # do the hankel transforms of the combined stack here
                r, Cr, Sr = cpp.radial_decompose_2D(comb_stack, 5)
                # append the hankel transform of the region to the list of all regions
                Cr_list.append(Cr)
                print("Number of peaks in region {:d}: {:d}.\n".format(reg,npks_reg))
                npks_list.append(npks_reg) # a list of the total number of clusters in each region
                binsize = (rad_in_Mpc / nbins).value
                y0, binned_r0 = cpp.bin_profile(r, Cr[:,0], rad_in_Mpc, binsize)
                Cr_binned = np.zeros((len(binned_r0),mmax))
                Cr_binned[:,0] = y0
                #binsize = rad_in_Mpc / len(binned_r0)
                for m in range(1,mmax):
                    y, binned_r = cpp.bin_profile(r, Cr[:,m], rad_in_Mpc, binsize)
                    Cr_binned[:,m] = y
                Cr_bin_list.append(Cr_binned)
        # if nothing has been found in any of these regions, skip it: only save file if there were peaks found
        if len(combstack_list) != 0:
            save_prof['binnedprof'] = Cr_bin_list
            save_prof['binned_r']   = binned_r
            save_prof['npks_list']  = npks_list
            save_prof['stacks']     = combstack_list
            save_prof['prof']       = Cr_list
            save_prof['rad_in_Mpc'] = rad_in_Mpc
            save_prof['region']     = reg
            if not os.path.exists(os.path.join(outpath, "temp_reg")):
                os.mkdir(os.path.join(outpath, "temp_reg"))
            save_file = open(os.path.join(outpath, "temp_reg", "{:s}_m0to{:d}_profiles.pkl".format(savestr, mmax)), "wb")
            pickle.dump(save_prof, save_file)
            save_file.close()
    else:
        print("Already done. moving on.")
        
def get_lensing_weights(cl_zlist, z_kernel, kernel):
        # get the weights for the lensing kernel
        # cl_zlist is the list of redshifts of the clusters
        # z_kernel is the list of redshifts of the kernel
        # kernel is the kernel itself
    weights = []
    for zbin in cl_zlist:
        # find weight at the center of the redshift bin
        # fix this later, this is the mean
        weights.append(1/np.mean(kernel[np.where((z_kernel<zbin[1]) & (z_kernel>zbin[0]))]))
    weights = u.Quantity(np.asarray(weights))
    a_cl = [1/(1+np.mean(zbin)) for zbin in cl_zlist]
    chi_cl = [cosmo.comoving_distance(np.mean(zbin)).value for zbin in cl_zlist]
    weights *= (1/np.asarray(a_cl)  * 1/(np.asarray(chi_cl)*u.Mpc) * 2*const.c**2/(8*np.pi *const.G))# units of surface mass densit
    print(weights[0]) # print units of weights
    return weights.to(u.Msun/u.Mpc**2).value

if errors:
    if nu_e_cuts:
        nreg  = 24
    else:
        nreg = 48
    from mpi4py import MPI
    # get the MPI ingredients     
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print("Rank = ", rank)
    nruns_local = nreg // size
    if rank == size-1:
        extras = nreg % size
    else:
        extras = 0

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
    #cutmin   = 10 # old way                                                                                                                                                                              
    cutmin   = 20 # new way                                                                                                                                                                               
    cutminstr = 'gt{:d}'.format(cutmin)
    cutmaxstr = ''
cutstr = '{:s}'.format(cut)+cutminstr+cutmaxstr

standard_stk_file = "standard_stackfile.ini"
standard_pk_file  = "standard_pkfile.ini"

if mode == 'ACTxDES':
    outpath     = "/mnt/scratch-lustre/mlokken/stacking/ACTxDES_paper2/"
    orient_mode = "maglim"
    ymap        = "/mnt/raid-cita/mlokken/data/act_ymaps/ilc_SZ_deproj_cib_yy_4096_hpx.fits" # CIB deprojected 
    
if mode == 'Buzzard':
    outpath     = "/mnt/scratch-lustre/mlokken/stacking/Buzzard_paper2/"
    orient_mode = "maglim"
    ymap        = "/mnt/raid-cita/mlokken/buzzard/ymaps/ymap_buzzard_fid_hpx.fits"
if mode == "Websky":
    outpath     = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/orient_tests/"
ymode       = os.path.split(ymap)[1][:-5]
stkpath = outpath + "orient_by_{:s}_{:d}/stacks".format(orient_mode, pct)
if not os.path.exists(stkpath):
    os.mkdir(stkpath)

dlist     = csf.dlist(minz=minz, maxz=maxz, slice_width=200)
if overlap:
    dlist_off = csf.dlist(minz=minz, maxz=maxz, slice_width=200, offset=100)
    dlist_tot = [None]*(len(dlist)+len(dlist_off))
    dlist_tot[::2] = dlist
    dlist_tot[1::2] = dlist_off
else:
    dlist_tot = dlist
print("List of slices of orientation:", dlist_tot)
if overlap:
    cl_dlist = [[dbin[0]+50,dbin[1]-50] for dbin in dlist_tot]
    print("List of slices of stacked clusters:", cl_dlist)

if errors:
    # non-interactive mode
    # set here before running
    
    # dlist is [[893, 993], [993, 1093], [1093, 1193], [1193, 1293],
    # [1293, 1393], [1393, 1493], [1493, 1593], [1593, 1693],
    # [1693, 1793], [1793, 1893], [1893, 1993], [1993, 2093],
    # [2093, 2193], [2193, 2293], [2293, 2393], [2393, 2493],
    # [2493, 2593], [2593, 2693], [2693, 2793], [2793, 2893],
    # [2893, 2993], [2993, 3093], [3093, 3193], [3193, 3293]]
    nlow_hi_bins = [[0,4], [6,10], [12,16], [18,22]]
    # nlow_hi_bins = [[0,22]]
    # corresponds to [[893, 1393], [1493, 1993], [2093, 2593], [2693, 3193]]

else:
    # Interactive mode
    nlow_hi_input = input("Input a space-separated list of the integers (start from 0) of the comoving distance slices (orientation) you would like to start and end each combined stack at. e.g. start1 end1 start2 end2 ....\n")
    nlow_hi_list  = nlow_hi_input.split(" ")
    print(nlow_hi_list)
    nlow_hi_bins  = [[int(nlow_hi_list[i]), int(nlow_hi_list[i+1])] for i in range(0, len(nlow_hi_list)-1, 2)]


save_prof = {}
if not errors:
    # now that all outputs are created, combine and stack all slices for each region
    for nbin in nlow_hi_bins:
        nlow = nbin[0]
        nhi  = nbin[1]
        
        if stack_y:
            mapstr = ymode
            consolidate_data_interactive(cl_dlist, dlist_tot, nlow, nhi, mapstr, cutstr, pt_selection_str, smth_str, pct, orient_mode)
                        
        if stack_kappa:
            mapstr = "DES_kappa"
            kernel = np.load("/home/mlokken/oriented_stacking/lensing_only_code/kernel_4.npy")
            z_kernel = np.load("/home/mlokken/oriented_stacking/lensing_only_code/z.npy")
            cl_zlist = z_at_value(cosmo.comoving_distance, cl_dlist*u.Mpc)
            weights = get_lensing_weights(cl_zlist, z_kernel, kernel)
            
            consolidate_data_interactive(cl_dlist, dlist_tot, nlow, nhi, mapstr, cutstr, pt_selection_str, smth_str, pct, orient_mode, weights=weights)

        if stack_galaxies:
            if mode=="ACTxDES":
                gmode = "DES"
            elif mode=="Buzzard":
                gmode = "Buzzard"
            if (nlow==0) & (nhi==4):
                mapstr = "{:s}_maglim_z_0pt20_0pt36".format(gmode)
            elif (nlow==6) & (nhi==10):
                mapstr = "{:s}_maglim_z_0pt36_0pt53".format(gmode)
            elif (nlow==12) & (nhi==16):
                mapstr = "{:s}_maglim_z_0pt53_0pt72".format(gmode)
            elif (nlow==18) & (nhi==22):
                mapstr = "{:s}_maglim_z_0pt72_0pt94".format(gmode)
        
            consolidate_data_interactive(cl_dlist, dlist_tot, nlow, nhi, mapstr, cutstr, pt_selection_str, smth_str, pct, orient_mode)
            

elif errors:
    for nbin in nlow_hi_bins:
        nlow = nbin[0]
        nhi  = nbin[1]

        if stack_y:
            mapstr = ymode
            collect_regions(nreg, cl_dlist, dlist_tot,  nlow, nhi, mapstr, cutstr, pt_selection_str, smth_str, pct, orient_mode)
        if stack_kappa:
            mapstr = "DES_kappa"
            kernel = np.load("/home/mlokken/oriented_stacking/lensing_only_code/kernel_4.npy")
            z_kernel = np.load("/home/mlokken/oriented_stacking/lensing_only_code/z.npy")
            cl_zlist = z_at_value(cosmo.comoving_distance, cl_dlist*u.Mpc)
            weights = get_lensing_weights(cl_zlist, z_kernel, kernel)
            collect_regions(nreg, cl_dlist, dlist_tot,  nlow, nhi, mapstr, cutstr, pt_selection_str, smth_str, pct, orient_mode, weights=weights)
        if stack_galaxies:
            if mode=="ACTxDES":
                gmode = "DES"
            elif mode=="Buzzard":
                gmode = "Buzzard"
            if (nlow==0) & (nhi==4):
                mapstr = "{:s}_maglim_z_0pt20_0pt36".format(gmode)
            elif (nlow==6) & (nhi==10):
                mapstr = "{:s}_maglim_z_0pt36_0pt53".format(gmode)
            elif (nlow==12) & (nhi==16):
                mapstr = "{:s}_maglim_z_0pt53_0pt72".format(gmode)
            elif (nlow==18) & (nhi==22):
                mapstr = "{:s}_maglim_z_0pt72_0pt94".format(gmode)
            collect_regions(nreg, cl_dlist, dlist_tot,  nlow, nhi, mapstr, cutstr, pt_selection_str, smth_str, pct, orient_mode)