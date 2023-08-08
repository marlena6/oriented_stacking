import numpy as np
import error_analysis_funcs as ef
import os
from astropy.cosmology import Planck18 as cosmo, z_at_value
from astropy.io import fits
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


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print("Rank = ", rank)

h = (cosmo.H0/100.).value
# mode = 'GRF'
# mode  = 'Buzzard'
mode = 'ACTxDES'
# mode   = 'Websky'

if mode=='Websky':
    cut = 'mass'
    cutmin = 1*10**13
    cutmax = 5*10**13
    #cutmax = None                                                                                                                                       
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
if mode == 'GRF':
    regstr = 'map'
else:
    regstr = 'reg'


if mode == 'Buzzard':
    nbins = 15
    # Bmode = 'GRF_comp'
    Bmode = 'basic'
if mode == 'ACTxDES':
    nbins = 12
if mode == 'Websky':
    nreg  = 48
    nbins = 60
if mode == 'GRF':
    nreg = 24
    nbins = 40

nslices = 1 # number of slices to be combined per stack
width = 200
#dmin  = 632
#dmax  = 5232

dmin  = 1032
dmax  = 2632

#dmin = 432
#dmax = 932

#zbins = [[0.20, 0.40], [0.4,0.55], [0.55,0.70], [0.70,0.85], [0.85,0.95]]
dbins_list = [(d, d + width*nslices) for d in range(dmin, dmax, width*nslices)]
if mode == 'GRF' or mode == 'Buzzard' and Bmode == 'GRF_comp':
    # overwrite some variables to make this version work
    zbins =[[0.15, 0.35], [0.35, 0.5], [0.5, 0.65]]
    dbins = [[int(cosmo.comoving_distance(z[0]).value),int(cosmo.comoving_distance(z[1]).value)] for z in zbins]
    dbins_list = dbins
    width = int(round((dbins[2][1]-dbins[0][0])))
smth_str  = '20pt0'
#smth_str = '13pt9'
#smth_str = '10pt0'
mmax = 5
plotsavepath = "/home/mlokken/oriented_stacking/plots/paper2/regions_plots/"

nu_e_cuts = True
if nu_e_cuts:
    pt_selection_str = "nugt2_egtpt3_"
    if mode == 'ACTxDES':
        nreg = 12
    elif mode == 'Buzzard':
        nreg = 24
else:
    pt_selection_str = ''
    if mode == 'ACTxDES':
        nreg = 48
    elif mode == 'Buzzard':
        nreg = 48
    if mode=='GRF' or (mode=='Buzzard' and Bmode=='GRF_comp'):
        pt_selection_str = 'allnu_alle_'
        # Martine update, remove later
        pt_selection_str = ''
if mode == 'GRF':
    add = '_gfield'
else:
    add = ''

addmap_str = 'ndmap_'
#''
#'kmap_'

if mode == 'ACTxDES':
    #outpath = "/mnt/scratch-lustre/mlokken/stacking/ACTxDES_paper2/"
    outpath = "/mnt/scratch-lustre/mlokken/stacking/D56_updated_redmagic_stacks/"
elif mode == 'Buzzard':
    outpath = "/mnt/scratch-lustre/mlokken/stacking/Buzzard_paper2/"

elif mode == 'Websky':
    if ndmap:
        outpath = "/mnt/scratch-lustre/mlokken/stacking/PeakPatch_halomap/orient_by_galmap_Mgt1e+13_msun_halos/"
    else:
        outpath = "/mnt/scratch-lustre/mlokken/stacking/PeakPatch_tSZ/orient_by_1pt5E12_to_1E15_msun_halos/"
        
elif mode == 'GRF':
    outpath = "/mnt/scratch-lustre/mlokken/stacking/GRF_buzzspec/"

random = False
if random:
    randstr = '_RANDOM'
else:
    randstr = ''
nruns_local = len(zbins)//size
if rank == size-1:
    extras = len(zbins) % size
else:
    extras = 0

for zbin in range(nruns_local+extras):
    if zsplit:
        zbin_local = zbins[zbin+rank*nruns_local]
        print(zbin_local)
        zmin_local  = zbin_local[0]
        zmax_local  = zbin_local[1]
        zstr_low  = str(zmin_local).replace(".","pt")
        zstr_hi   = str(zmax_local).replace(".","pt")
        dlow = cosmo.comoving_distance(zmin_local).value
        dhi  = cosmo.comoving_distance(zmax_local).value
        binstr = "z_{:s}_{:s}".format(zstr_low, zstr_hi)
    else:
        dbin_local = dbins[zbin+rank*nruns_local]
        dlow = dbin_local[0]
        dhi  = dbin_local[1]
        binstr = "{:d}_{:d}Mpc".format(dlow, dhi)
    # now that all outputs are created, combine and stack all slices for each region

    combined_stacks = []
    combined_profs  = []
    npks_allreg = []
    save_prof = {}

    for reg in range(nreg):
        reg_path  = os.path.join(outpath,str(reg))
        stack_list = []
        npks_reg_list = []
        skip_list = []
        npks_reg = 0
        hankel_arrays = []
        print("Region",reg)
        file_root = "{:s}_{:s}{:s}_reg{:d}_{:s}_{:s}{:s}{:s}{:s}_100pctmaglim".format(mode, addmap_str, cutstr, reg, binstr, pt_selection_str, smth_str, randstr, add)
        print(file_root)
        stackfile = os.path.join(reg_path,"{:s}_stk.fits".format(file_root))
        if addmap_str == '':
            pkfile = os.path.join(reg_path,"{:s}_pks.fits".format(file_root))
        else:
            pkfile = os.path.join(reg_path,"{:s}_pks.fits".format(file_root.replace(addmap_str,'')))
        if os.path.exists(stackfile):
            hdr, img, npks = cpp.get_img(stackfile, pkfile)
            stack_list.append(img)
            npks_reg += npks
            npks_reg_list.append(npks)
            hankel_ms = []
            for m in range(mmax):
                hankelfile = os.path.join(reg_path,"{:s}_stk_HankelTransform_m{:d}.txt".format(file_root, m))
                hankel = np.loadtxt(hankelfile)
                r, Cr, Sr = hankel[:,0], hankel[:,1], hankel[:,2]
                hankel_ms.append(Cr)
            hankel_ms = np.asarray(hankel_ms).T
            hankel_arrays.append(hankel_ms)
        else:
            print("{:s} doesn't exist.".format(stackfile))
            stack_list.append(0)
            npks_reg_list.append(0)
            hankel_arrays.append(0)

        if len(skip_list) == 0:
            skip_list = None
        mid = (dhi + dlow)/2. * u.Mpc
        z = z_at_value(cosmo.comoving_distance, mid)
        mpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(z).to(u.Mpc / u.arcmin)
        inifile = fits.open(stackfile)
        hdr = inifile[0].header
        radius = float(hdr['RADIUS'].replace('=',''))
        res    = float(hdr['RES'].replace('=',''))
        arcmin_per_pix = (radius*u.deg/res).to(u.arcmin)
        rad_in_Mpc = hankel_ms.shape[0] * arcmin_per_pix * mpc_per_arcmin
        comb_prof = hankel_ms
        comb_stack = img
        print("Number of peaks in region {:d}: {:d}.\n".format(reg,npks_reg))
        npks_allreg.append(npks_reg)
        binsize = (rad_in_Mpc / nbins).value
        print("binning with a size of {}".format(binsize))
        y0, binned_r0 = cpp.bin_profile(comb_prof[:,0], rad_in_Mpc, binsize)
        all_ms = np.zeros((len(binned_r0),mmax))
        all_ms[:,0] = y0
        for m in range(1,mmax):
            y, binned_r = cpp.bin_profile(comb_prof[:,m], rad_in_Mpc, binsize)
            print(m, len(y))
            all_ms[:,m] = y
        save_prof['binnedprof_{:s}{:d}'.format(regstr,reg)] = all_ms
        save_prof['npks_{:s}{:d}'.format(regstr,reg)] = npks_reg
        save_prof['profs_{:s}{:d}'.format(regstr,reg)] = comb_prof
        combined_stacks.append(comb_stack)
        combined_profs.append(comb_prof)
        print('end of region {:d}'.format(reg))

    # all-region combined stack
    all_stack = np.zeros(combined_stacks[0].shape)
    all_prof  = np.zeros(combined_profs[0].shape)
    for reg in range(nreg):
        all_stack += combined_stacks[reg] * npks_allreg[reg]
        all_prof  += combined_profs[reg] * npks_allreg[reg]
    all_stack /= sum(npks_allreg)
    all_prof /= sum(npks_allreg)
    fig = plt.figure(figsize=[8,5])
    smoothplot = plt.imshow(np.flipud(ndimage.gaussian_filter(all_stack, sigma=8)), cmap='afmhot')
    mpc_per_pix = rad_in_Mpc / (all_stack.shape[0]//2)
    locs = []
    for i in range(9):
        locs.append(i/8. * img.shape[0])
    labels = []
    for loc in locs:
        labels.append(round((loc-img.shape[0]//2) * mpc_per_pix.value))
    plt.xlabel("x [cMpc]")
    plt.ylabel("y [cMpc]")
    plt.xticks(locs, labels)
    plt.yticks(locs, labels)
    plt.title("{:.0f}-{:.0f} Mpc, {:d} pts".format(dlow, dhi, sum(npks_allreg)))
    cbar = fig.colorbar(smoothplot)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.set_label('Compton-$y$')
    cbar.update_ticks()
    plt.tight_layout()
    savestr = "{:s}_{:s}{:s}_z_{:s}_{:s}_{:s}Mpc_{:s}smth_{:d}{:s}".format(mode, addmap_str, cutstr, zstr_low, zstr_hi, smth_str, pt_selection_str, nreg, regstr)
    plt.savefig(plotsavepath + savestr+"all{:s}_combined_stack{:s}_100pctmaglim.png".format(regstr,randstr))
    plt.clf()

    np.savetxt(os.path.join(outpath,savestr+"all{:s}_combined_stack{:s}_100pctmaglim.txt".format(regstr,randstr)), all_stack)

    # r, Cr, Sr = cpp.radial_decompose_2D(f=all_stack, mmax=mmax)
    y0, binned_r0 = cpp.bin_profile(all_prof[:,0], rad_in_Mpc, binsize)
    all_ms = np.zeros((len(binned_r0), mmax))
    all_ms[:,0] = y0
    for m in range(1,mmax):
        y, binned_r = cpp.bin_profile(all_prof[:,m], rad_in_Mpc, binsize)
        all_ms[:,m] = y
        # plt.plot(binned_r, save_prof['binnedprof_all_reg'][m,:], 'k', label='Mean profile')
        # for reg in range(nreg):
        #     plt.plot(binned_r, save_prof['reg{:d}'.format(reg)][m,:], 'gray', alpha=0.5)
        # plt.xlabel("r [Mpc]")
        # plt.ylabel(r"Compton-$y$")
        # plt.title("m = {:d} component, {:d} regions".format(m, nreg))
        # plt.legend()
        # plt.savefig("/home/mlokken/actxdes_stacking/plots/regions_plots/{:s}_meq{:d}_allprofs.png".format(savestr, m))
        # plt.clf()
    save_prof['binnedprof_all{:s}'.format(regstr)] = all_ms
    save_prof['binned_r'] = binned_r
    save_prof['prof_all{:s}'.format(regstr)] = all_prof
    save_prof['npks_total'] = sum(npks_allreg)
    save_file = open(os.path.join(outpath, "{:s}_m0to{:d}_profiles{:s}_100pctmaglim.pkl".format(savestr, mmax, randstr)), "wb")
    print(os.path.join(outpath, "{:s}_m0to{:d}_profiles_100pctmaglim.pkl".format(savestr,mmax,randstr)))
    pickle.dump(save_prof, save_file)
    save_file.close()
