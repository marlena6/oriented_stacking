import os
from astropy.io import fits
import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from astropy.cosmology import Planck18 as cosmo, z_at_value
from PIL import Image
import math
import coop_setup_funcs as csf

# from numba import jit

def average_regions(splits, npks_list, spatial_weights=None):
    # get the average of regions with different number of stacked peaks per region
    # splits can either be an array of profiles with the 0 axis corresponding to each 'observation'
    # or splits can be an array of images with the 0 axis corresponding to each 'observation'
    if spatial_weights is None:
        spatial_weights = np.ones(len(npks_list))
    weights = npks_list / np.average(npks_list) * spatial_weights
    return np.average(np.asarray(splits), axis=0, weights=weights)



def bin_profile(r, Cr_m, rad_in_mpc, binsize): # binsize in Mpc
    # r is in Mpc
    
    npix    = len(Cr_m)
    mpc_per_pix = rad_in_mpc / (npix-1)
    pix_per_bin = int(math.floor((binsize / mpc_per_pix)))
    binned_prof = []
    binned_r    = []
    for i in range(0,npix-pix_per_bin,pix_per_bin):
        binned_prof.append(np.mean(Cr_m[i:i+pix_per_bin]))
        binned_r.append(np.mean(r[i:i+pix_per_bin]))
    #r = [i + 0.5 for i in range(len(binned_prof))]
    #step_in_mpc = rad_in_mpc/len(r)
    return binned_prof, binned_r

def cormat(covmat):
    cormat = np.zeros(covmat.shape)
    for i in range(covmat.shape[0]):
        for j in range(covmat.shape[1]):
            corcoeff = covmat[i,j]/np.sqrt(covmat[i,i]*covmat[j,j])
            cormat[i,j] = corcoeff
    return(cormat)

def covariances(y_list,weights,nreg):
    y_array = np.asarray(y_list)
    covmat = np.cov(y_array.T, aweights=weights)/nreg
    correl_mat = cormat(covmat)
    return(covmat, correl_mat)

def get_img(stackfile, pkfile):
    stack = fits.open(stackfile)
    hdr = stack[0].header
    img = stack[0].data
    stack.close()
    pks    = fits.open(pkfile)
    npeaks = len(pks[0].data)
    pks.close()
    return hdr, img, npeaks

def getprofs(pkl,nreg,m):
    npks_reg = pkl['npks_list']
    ys_all_regions_full = []
    ys_all_regions_binned = []
    for reg in range(nreg):
        y_binned = pkl['binnedprof'][reg][:,m]
        y = pkl['prof'][reg][:,m]
        ys_all_regions_binned.append(y_binned)
        ys_all_regions_full.append(y)
    weights = npks_reg / np.average(npks_reg)
    covmat_binned, cormat_binned = covariances(ys_all_regions_binned, weights, nreg)
    covmat_full, cormat_full = covariances(ys_all_regions_full, weights, nreg)
    errors_binned = np.sqrt(np.diag(covmat_binned))
    errors_full = np.sqrt(np.diag(covmat_full))
    return(covmat_full, cormat_full, errors_full, covmat_binned, cormat_binned, errors_binned, np.sum(npks_reg))

def getprofs_sims(pkl,nmap,m):
    y_mean        = pkl['prof_allmap'][:,m]
    y_mean_binned = pkl['binnedprof_allmap'][:,m]
    binned_r      = pkl['binned_r']
    npks_reg = []
    npks_reg      = np.asarray([pkl['npks_map{:d}'.format(reg)] for reg in range(nmap)])
    npks_reg      = np.asarray(npks_reg)
    npks_tot      = pkl['npks_total']
    ys_all_regions_full = []
    ys_all_regions_binned = []
    for reg in range(nmap):
        y_binned = np.copy(pkl['binnedprof_map{:d}'.format(reg)][:,m])
        y = np.copy(pkl['profs_map{:d}'.format(reg)][:,m])
        if m==0:
            tail = np.average(y[200:])
            y_binned -= tail
            y -= tail
        ys_all_regions_binned.append(y_binned)
        ys_all_regions_full.append(y)
    weights = npks_reg / np.average(npks_reg)
    covmat_binned, cormat_binned = covariances(ys_all_regions_binned, weights, 1)
    covmat_full, cormat_full     = covariances(ys_all_regions_full, weights, 1)
    errors_binned = np.sqrt(np.diag(covmat_binned))
    errors_full   = np.sqrt(np.diag(covmat_full))
    return(np.arange(len(y_mean)), y_mean, covmat_full, cormat_full, errors_full, binned_r, y_mean_binned, covmat_binned, cormat_binned, errors_binned, npks_tot)

def getbinnedprofs_custombins(pkl, nreg, pix_bins, m, mpc_per_pix, covmat_full=None): #covmat_full is alternative covariance matrix from different error estimation method
    y_mean        = pkl['prof_allreg']
    npks_reg      = np.asarray([pkl['npks_reg{:d}'.format(reg)] for reg in range(nreg)])
    npks_tot      = pkl['npks_total']
    binned_r      = np.asarray([(pix_bins[i]+pix_bins[i+1])/2. for i in range(len(pix_bins)-1)]) * mpc_per_pix
    weights = npks_reg / np.average(npks_reg)
    y_mean_binned = [np.mean(y_mean[:,m][pix_bins[i]:pix_bins[i+1]]) for i in range(len(pix_bins)-1)]
    if covmat_full is None:
        ys_all_regions_full = []
        for reg in range(nreg):
            y = np.copy(pkl['profs_reg{:d}'.format(reg)][:,m])
            if m==0:
                ytail = np.average(y[200:])
                y -= ytail
            ys_all_regions_full.append(y)
        covmat_full, cormat_full = covariances(ys_all_regions_full, weights, nreg)
    errors_full = np.sqrt(np.diag(covmat_full))
    ys_all_regions_binned = []
    for reg in range(nreg):
        y = np.copy(pkl['profs_reg{:d}'.format(reg)][:,m])
        if m==0:
            ytail = np.average(y[200:])
            y -= ytail
        binweights = [1/errors_full[pix_bins[i]:pix_bins[i+1]] for i in range(len(pix_bins)-1)]
        ys_all_regions_binned.append([np.average(y[pix_bins[i]:pix_bins[i+1]], weights=binweights[i]) for i in range(len(pix_bins)-1)])
    covmat_binned, cormat_binned = covariances(ys_all_regions_binned, weights, nreg)
    errors_binned = np.sqrt(np.diag(covmat_binned))
    return(binned_r, y_mean_binned, covmat_binned, cormat_binned, errors_binned, npks_tot)
    
def getbinnedprofs_custombins_sims(pkl, nmap, pix_bins, m, mpc_per_pix, covmat_full=None):
    y_mean        = pkl['prof_allmap'][:,m]
    npks_reg      = np.asarray([pkl['npks_map{:d}'.format(reg)] for reg in range(nmap)])
    npks_tot      = pkl['npks_total']
    y_mean_binned = [np.mean(y_mean[pix_bins[i]:pix_bins[i+1]]) for i in range(len(pix_bins)-1)]
    binned_r      = np.asarray([(pix_bins[i]+pix_bins[i+1])/2. for i in range(len(pix_bins)-1)]) * mpc_per_pix
    if covmat_full is None:
        ys_all_regions_full = []
        for reg in range(nreg):
            y = np.copy(pkl['profs_map{:d}'.format(reg)][:,m])
            if m==0:
                ytail = np.average(y[200:])
                y -= ytail
            ys_all_regions_full.append(y)
        covmat_full, cormat_full = covariances(ys_all_regions_full, weights, nreg)
    errors_full = np.sqrt(np.diag(covmat_full))
    ys_all_regions = []
    for reg in range(nmap):
        y = np.copy(pkl['profs_map{:d}'.format(reg)][:,m])
        if m==0:
            ytail = np.average(y[200:])
            y -= ytail
        binweights = [1/errors_full[pix_bins[i]:pix_bins[i+1]] for i in range(len(pix_bins)-1)]
        ys_all_regions.append([np.average(y[pix_bins[i]:pix_bins[i+1]], weights=binweights[i]) for i in range(len(pix_bins)-1)])

    weights = npks_reg / np.average(npks_reg)
    covmat, cormat = covariances(ys_all_regions, weights, 1)
    errors = np.sqrt(np.diag(covmat))
    return(binned_r, y_mean_binned, covmat, cormat, errors, npks_tot)

def peakinfo_radec(filename):
    peakfile = fits.open(filename)
    peakinfo = peakfile[0].data
    peakfile.close()
    rot_angle = peakinfo[:,3]
    theta,phi = peakinfo[:,1], peakinfo[:,2]
    dec = []
    ra  = []
    for i in range(len(theta)):
        dec.append(csf.ThetaPhitoRaDec(theta[i],phi[i])[1])
        ra.append(csf.ThetaPhitoRaDec(theta[i],phi[i])[0])
    ra = np.asarray(ra)
    dec = np.asarray(dec)
    return (rot_angle,ra,dec)

def peakinfo_thetaphi(filename):
    peakfile = fits.open(filename)
    peakinfo = peakfile[0].data
    peakfile.close()
    rot_angle = peakinfo[:,3]
    theta,phi = peakinfo[:,1], peakinfo[:,2]
    return (rot_angle,theta,phi)

def convert_xperpix_phys_units_hp(nside, redshift, cosmology):
    # takes an nside, redshift, and astropy cosmology object
    import healpy as hp
    from astropy import units as u
    pix_area_sq_deg = hp.nside2pixarea(nside, degrees=True)*(u.deg)**2
    sqMpc_per_sqdeg = (cosmology.kpc_comoving_per_arcmin(redshift).to(u.Mpc / u.deg))**2
    pix_area_sq_Mpc = sqMpc_per_sqdeg * pix_area_sq_deg
    return pix_area_sq_deg, pix_area_sq_Mpc # number of square degrees per pixel, number of square Mpc per pixel

def ngal_convert(img, mid_slice_distance, arcmin_per_pix):
    z = z_at_value(cosmo.comoving_distance, mid_slice_distance * u.Mpc)
    mpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(z).to(u.Mpc / u.arcmin)
    len_in_mpc = img.shape[0] * arcmin_per_pix * mpc_per_arcmin # the length of the first image in comoving Mpc
    wid_in_mpc = img.shape[1] * arcmin_per_pix * mpc_per_arcmin # the length of the first image in comoving Mpc
    sq_pix  = img.shape[0]*img.shape[1]
    sq_mpc  = len_in_mpc.value*wid_in_mpc.value
    img = img * sq_pix/sq_mpc
    return(img, sq_pix/sq_mpc)

def ngal_convert_prof(ng_per_pix, mid_slice_distance, arcmin_per_pix):
    z = z_at_value(cosmo.comoving_distance, mid_slice_distance * u.Mpc)
    mpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(z).to(u.Mpc / u.arcmin)
    img_side_len   = len(ng_per_pix)*2+1
    sq_pix         = img_side_len**2
    len_in_mpc     = img_side_len * arcmin_per_pix * mpc_per_arcmin
    sq_Mpc         = (len_in_mpc**2).value
    # ngal per mpc^2 = ngal per pixel * 1 (pixel) / Mpc^2 per pixel 
    ng_per_mpc = ng_per_pix * sq_pix/sq_Mpc
    return(ng_per_mpc)

def radial_decompose_2D(f, mmax):
        # f is numpy image array                                                                                                                                                                                  
        # mmax is maximum m for decomposition (maximally 10)                                                                                                                                                     
        n = int(f.shape[0] // 2)
        nsteps = n * 20
        dtheta = 2*np.pi/nsteps
        Cr = np.zeros((n-1, mmax))
        Sr = np.zeros((n-1, mmax))

        for i in range(1, n):
                r = float(i)
                for j in range(nsteps):
                        # print(j)                                                                                                                                                                                
                        theta = dtheta * j
                        # print(theta)                                                                                                                                                                            
                        rx    = r*np.cos(theta)
                        ry    = r*np.sin(theta)
                        ix    = min(math.floor(rx), n-1)
                        iy    = min(math.floor(ry), n-1)
                        rx    = rx - ix
                        ry    = ry - iy
                        ix    = ix + n # different from Fortran COOP version -- indexing middle of array                                                                                                          
                        iy    = iy + n
                        fv    = (1-rx)*(f[iy, ix]*(1-ry) + f[iy+1, ix]*ry) + rx * ( f[iy, ix+1]*(1-ry) + f[iy+1, ix+1]*ry)
                        Cr[i-1,0] += fv
                        for m in range(1, mmax):
                                Cr[i-1,m] = Cr[i-1,m] + fv * np.cos(m*theta)
                                Sr[i-1,m] = Sr[i-1,m] + fv * np.sin(m*theta)
                
        Cr[0:n-1,0] = Cr[0:n-1, 0]/nsteps
        Cr[0:n-1,1:mmax] =  Cr[0:n-1,1:mmax] * (2./nsteps)
        Sr[0:n-1,1:mmax] =  Sr[0:n-1,1:mmax] * (2./nsteps)
        return(np.arange(1,n), Cr, Sr)

#@jit
def radial_decompose_2D_fast(f, mmax):
    # this doesn't work yet
    # f is numpy image array
    # mmax is maximum m for decomposition (maximally 10)
    n = int(f.shape[0] // 2)
    nsteps = n * 20
    dtheta = 2*np.pi/nsteps
    Cr = np.zeros((n, mmax))
    Sr = np.zeros((n, mmax))
    
    for i in range(0, n):
        r = float(i)
        j = np.arange(0, nsteps, dtype=int)
        # print(j)                                                                                                                                                                                
        theta = dtheta * j
        # print(theta)                                                                                                                                                                            
        rx    = r*np.cos(theta)
        ry    = r*np.sin(theta)
        ix    = np.minimum(np.floor(rx), n-1).astype(int)
        iy    = np.minimum(np.floor(ry), n-1).astype(int)
        rx    = rx - ix
        ry    = ry - iy
        ix    = ix + n # different from Fortran COOP version -- indexing middle of array                                                                                                          
        iy    = iy + n
        # this part appears not to work without a for loop
        for j in range(nsteps):
            fv    = (1-rx[j])*(f[iy[j], ix[j]]*(1-ry[j]) + f[iy[j]+1, ix[j]]*ry[j]) + rx[j] * ( f[iy[j], ix[j]+1]*(1-ry[j]) + f[iy[j]+1, ix[j]+1]*ry[j])
            Cr[i,0] += fv
        for m in range(1, mmax):
            Cr[i,m] = Cr[i,m] + fv * np.cos(m*theta)
            Sr[i,m] = Sr[i,m] + fv * np.sin(m*theta)
            # print("cos2theta ", np.cos(2*theta))                                                                                                                                                    
        # print("fv", fv)                                                                                                                                                                         

        Cr[0:n,0] = Cr[0:n, 0]/nsteps
        Cr[0:n,1:mmax] =  Cr[0:n,1:mmax] * (2./nsteps)
        Sr[0:n,1:mmax] =  Sr[0:n,1:mmax] * (2./nsteps)
        return(np.arange(n), Cr, Sr)


def stack_multi(stack_list, npks_list, d_low, slice_width, arcmin_per_pix, nstack=None, skip_list=None, ngal=False):
    # stack_list is list of arrays to be stacked
    # d_low is distance to nearest end of closest stack [cMpc]
    # slice_width is radial width of slices [cMpc]

    d_low = int(d_low)
    slice_width = int(slice_width)
    if skip_list == None:
        base_shape = stack_list[0].shape
    else:
        for l in range(len(stack_list)):
            if l not in skip_list:
                base_shape = stack_list[l].shape
                break
    
    rad_in_pix = base_shape[0]//2 # images are always odd number of pixels wide, so this is # of pixels starting from 1st from center
    im_array = np.zeros(base_shape)
    if nstack==None:
        nstack = len(stack_list)
    c = 0
    for i in range(d_low, d_low + nstack * slice_width, slice_width):
        uplim = i + slice_width
        mid = (i + uplim)/2. * u.Mpc
        z = z_at_value(cosmo.comoving_distance, mid)
        mpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(z).to(u.Mpc / u.arcmin)
        if c==0:
            base_scale = mpc_per_arcmin
            zoom_level = 1.
            if ((skip_list is not None) and (c not in skip_list)) or skip_list == None:
                if ngal:
                    im_array += (ngal_convert(stack_list[c] * npks_list[c], mid.value, arcmin_per_pix))[0]
                else:
                    im_array += stack_list[c] * npks_list[c]
            # if there is a skip list and 0 is in it, do nothing (array remains 0s)
            rad_in_mpc = rad_in_pix * arcmin_per_pix * mpc_per_arcmin # the radius of the first image in comoving Mpc
            
        else:
            # for images centered on further distances, the same radius in degrees
            # corresponds to a larger transverse physical radius. Trim these images
            # to the same radius in Mpc as the first, then resize the array.

            zoom_level = mpc_per_arcmin/base_scale
            xmax = int(rad_in_pix + 1/(zoom_level)*rad_in_pix)
            xmin = int(rad_in_pix - 1/(zoom_level)*rad_in_pix)

            if ((skip_list != None) and (c not in skip_list)) or skip_list == None:
                if ngal:
                    img_to_add = (ngal_convert(stack_list[c][xmin:xmax, xmin:xmax] * npks_list[c], mid.value, arcmin_per_pix))[0]
                else:
                    img_to_add = stack_list[c][xmin:xmax, xmin:xmax] * npks_list[c]
                pil_img    = Image.fromarray(img_to_add)
                pil_img_rs = pil_img.resize(base_shape)
                resized    = np.array(pil_img_rs.getdata()).reshape(pil_img_rs.size)
                im_array += resized
            else:
                print("Skipping array {:d} in the stack, assuming there's missing data in {:d} to {:d} Mpc slice.\n".format(c, i, uplim))
        c += 1
    return im_array / sum(npks_list), rad_in_mpc

def stack_multi_same_csize(stack_list, npks_list, d_low, slice_width, arcmin_per_pix, nstack=None, skip_list=None, ngal=False, base_shape=None, multiplier=None):
    # stack_list is list of arrays to be stacked
    # d_low is distance to nearest end of closest stack [cMpc]
    # slice_width is radial width of slices [cMpc]
    d_low = int(d_low)
    slice_width = int(slice_width)
    if base_shape is None:
        if skip_list == None:
            base_shape = stack_list[0].shape
        else:
            for l in range(len(stack_list)):
                if l not in skip_list:
                    base_shape = stack_list[l].shape
                    break
    if multiplier is None:
        multiplier = np.full(len(stack_list), 1.)
    rad_in_pix = base_shape[0]//2
    im_array = np.zeros(base_shape)
    if nstack==None:
        nstack = len(stack_list)
        print("Number of slices to stack", nstack)
    c = 0
    dlist = [d for d in range(d_low, d_low + nstack * slice_width, slice_width)]
    print(dlist)
    for i in dlist:
        uplim = i + slice_width
        mid = (i + uplim)/2. * u.Mpc
        z = z_at_value(cosmo.comoving_distance, mid)
        mpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(z).to(u.Mpc / u.arcmin)
        if c==0:
            if ((skip_list is not None) and (c not in skip_list)) or skip_list == None:
                if ngal:
                    im_array += (ngal_convert(stack_list[c] * npks_list[c] * multiplier[c], mid.value, arcmin_per_pix))[0]
                else:
                    im_array += stack_list[c] * npks_list[c] * multiplier[c]
            # if there is a skip list and 0 is in it, do nothing (array remains 0s)
            rad_in_mpc = rad_in_pix * arcmin_per_pix * mpc_per_arcmin # the radius of the first image in comoving Mpc
            
        else:
            # for images centered on further distances, the same radius in degrees
            # corresponds to a larger transverse physical radius. Trim these images
            # to the same radius in Mpc as the first, then resize the array.
            
            if ((skip_list != None) and (c not in skip_list)) or skip_list == None:
                if ngal:
                    img_to_add = (ngal_convert(stack_list[c] * npks_list[c] * multiplier[c], mid.value, arcmin_per_pix))[0]
                else:
                    img_to_add = stack_list[c] * npks_list[c] * multiplier[c]
                pil_img    = Image.fromarray(img_to_add)
                pil_img_rs = pil_img.resize(base_shape)
                resized    = np.array(pil_img_rs.getdata()).reshape(pil_img_rs.size)
                im_array += resized
            else:
                print("Skipping array {:d} in the stack, assuming there's missing data in {:d} to {:d} Mpc slice.\n".format(c, i, uplim))
        c += 1
    return im_array / sum(npks_list), rad_in_mpc

def hankel_multi(hankel_list, npks_list, d_low, slice_width, arcmin_per_pix, nstack=None, skip_list=None, ngal=False):
    # a function for combining the same multipole moment radial profile for successive stacks in distance when the angular size of the stacks is the same 
    from scipy.interpolate import interp1d
    # hankel_list is list of arrays with shape (image radius in pixels, mmax)
    # d_low is distance to nearest end of closest stack [cMpc]
    # slice_width is radial width of slices [cMpc]

    d_low = int(d_low)
    slice_width = int(slice_width)
    if skip_list == None:
        rad_in_pix = hankel_list[0].shape[0]
        mmax = hankel_list[0].shape[1]
    else:
        for l in range(len(hankel_list)):
            if l not in skip_list:
                rad_in_pix = hankel_list[l].shape[0]
                mmax = hankel_list[l].shape[1]
                break
    
    if nstack==None:
        nstack = len(hankel_list)
    c = 0
    print("mmax = {:d}".format(mmax))
    hankel_array = np.zeros((rad_in_pix, mmax))
    for i in range(d_low, d_low + nstack * slice_width, slice_width):
        uplim = i + slice_width
        mid = (i + uplim)/2. * u.Mpc
        z = z_at_value(cosmo.comoving_distance, mid)
        mpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(z).to(u.Mpc / u.arcmin)
        if c==0:
            base_scale = mpc_per_arcmin
            zoom_level = 1.
            if ((skip_list is not None) and (c not in skip_list)) or skip_list == None:
                for m in range(mmax):
                    if ngal:
                        hankel_array[:,m] += ngal_convert_prof((hankel_list[c][:,m] * npks_list[c]), mid.value, arcmin_per_pix)
                    else:
                        hankel_array[:,m] += hankel_list[c][:,m] * npks_list[c]
            # if there is a skip list and 0 is in it, do nothing (array remains 0s)
            rad_in_mpc = rad_in_pix * arcmin_per_pix * mpc_per_arcmin # the radius of the first image in comoving Mpc
        else:
            # for images centered on further distances, the same radius in degrees
            # corresponds to a larger transverse physical radius. Trim these profiles
            # to the same radius in Mpc as the first, then resize the array.
            zoom_level = mpc_per_arcmin/base_scale
            xmax = int(1/(zoom_level)*rad_in_pix)
            if ((skip_list != None) and (c not in skip_list)) or skip_list == None:
                for m in range(mmax):
                    prof_to_add = hankel_list[c][:,m][:xmax] * npks_list[c]
                    prof_func   = interp1d(np.arange(len(prof_to_add)), prof_to_add)
                    resized     = prof_func(np.linspace(0,len(prof_to_add)-1,rad_in_pix)) #ML: replaced 240 with rad_in_pix, should be same but more general
                    if ngal:
                        hankel_array[:,m] += ngal_convert_prof(resized, mid.value, arcmin_per_pix)
                    else:
                        hankel_array[:,m]  += resized
            else:
                print("Skipping array {:d} in the stack, assuming there's missing data in {:d} to {:d} Mpc slice.\n".format(c, i, uplim))
        c += 1
    return hankel_array / sum(npks_list), rad_in_mpc

def hankel_multi_same_csize(hankel_list, npks_list, d_low, slice_width, arcmin_per_pix, nstack=None, skip_list=None, ngal=False, multiplier=None):
    # a function for combining the same multipole moment radial profile for successive stacks in distance when the comoving size of the stacks is the same
    from scipy.interpolate import interp1d
    # hankel_list is list of arrays with shape (image radius in pixels, mmax)
    # d_low is distance to nearest end of closest stack [cMpc]
    # slice_width is radial width of slices [cMpc]

    d_low = int(d_low)
    slice_width = int(slice_width)
    if skip_list == None:
        rad_in_pix = hankel_list[0].shape[0]
        mmax = hankel_list[0].shape[1]
    else:
        for l in range(len(hankel_list)):
            if l not in skip_list:
                rad_in_pix = hankel_list[l].shape[0]
                mmax = hankel_list[l].shape[1]
                break
    if nstack==None:
        nstack = len(hankel_list)
    print("mmax = {:d}".format(mmax))
    hankel_array = np.zeros((rad_in_pix, mmax))
    dlist = [d for d in range(d_low, d_low + nstack * slice_width, slice_width)]
    print(dlist)
    c = 0
    for i in dlist:
        uplim = i + slice_width
        mid = (i + uplim)/2. * u.Mpc
        z = z_at_value(cosmo.comoving_distance, mid)
        mpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(z).to(u.Mpc / u.arcmin)
        if c==0:
            if ((skip_list is not None) and (c not in skip_list)) or skip_list == None:
                for m in range(mmax):
                    if ngal:
                        hankel_array[:,m] += ngal_convert_prof((hankel_list[c][:,m] * npks_list[c]*multiplier[c]), mid.value, arcmin_per_pix)
                    else:
                        hankel_array[:,m] += hankel_list[c][:,m] * npks_list[c] * multiplier[c]
            # if there is a skip list and 0 is in it, do nothing (array remains 0s)
            rad_in_mpc = rad_in_pix * arcmin_per_pix * mpc_per_arcmin # the radius of the first image in comoving Mpc
        else:
            if ((skip_list != None) and (c not in skip_list)) or skip_list == None:
                for m in range(mmax):
                    prof_to_add = hankel_list[c][:,m] * npks_list[c]
                    prof_func   = interp1d(np.arange(len(prof_to_add)), prof_to_add)
                    resized     = prof_func(np.linspace(0,len(prof_to_add)-1,rad_in_pix))
                    if ngal:
                        hankel_array[:,m] += ngal_convert_prof(resized, mid.value, arcmin_per_pix)
                    else:
                        hankel_array[:,m]  += resized
            else:
                print("Skipping array {:d} in the stack, assuming there's missing data in {:d} to {:d} Mpc slice.\n".format(c, i, uplim))
        c += 1
    return hankel_array / sum(npks_list), rad_in_mpc

class Stack_object:
    # an object to be loaded in from a file of an errors run
    # Not using any Astropy Quantities in this class because they cause bugs
    def __init__(self, rad_in_Mpc, avg_img=None, avg_profiles=None, img_splits=None, profile_splits=None, Npks_splits=None):
        # Img is an array of shape (img_side_len, img_side_len)
        # avg_profiles is an array of shape (m_max, img_side_len//2)
        # Img_splits is an array of shape (N_splits, img_side_len, img_side_len)
        # Profile_splits is a list of length m_max, each element of list is array of shape (N_splits, img_side_len//2)
        # Npks_splits is an array of shape (N_splits,)
        # Rad_in_Mpc is the radius of the stack image in Mpc
        
        # begin with some checks
        if avg_img is None and img_splits is None:
            print("Must provide either img or img_splits.")
            return
        if avg_img is not None and type(avg_img) != np.ndarray:
            print("img must be a numpy array.")
            return
        if avg_profiles is not None and type(avg_profiles) not in [np.ndarray, list]:
            print("profiles must be a numpy array or list.")
            return
        # Convert some lists to arrays if necessary
        if img_splits is not None and type(img_splits) not in [np.ndarray, list]:
            print("img_splits must be a numpy array or list.")
            return
        if profile_splits is not None and type(profile_splits) not in [np.ndarray, list]:
            print("profile_splits must be a numpy array or list.")
            return
        if type(img_splits)  == list:
            img_splits = np.asarray(img_splits)
        if type(Npks_splits) == list:
            Npks_splits = np.asarray(Npks_splits)
        
        self.__has_splits__ = False
        if img_splits is not None:
            self.__has_splits__ = True
        
        self.avg_img         = avg_img # stack image
        self.img_splits      = img_splits # stack images in splits. The weighted average of these should be the full stack image
        self.profile_splits  = profile_splits # unbinned multipole profiles in splits
        self.rad_in_Mpc      = rad_in_Mpc # radius of the stack image in Mpc
        self.Npks_splits     = Npks_splits # number of peaks in each split
        self.avg_profiles    = avg_profiles # list of the unbinned average profiles for each multipole moment m. Length m_max, each element shape (n_bins,)
        
        if self.__has_splits__:
            self.Nsamples = len(img_splits) # number of samples
            self.split_wgts = self.Npks_splits / np.average(self.Npks_splits)
            self.mmax = len(profile_splits) # maximum multipole moment
        if self.avg_profiles is None:
            self.avg_profiles = []
            for m,profsplits in enumerate(self.profile_splits):
                self.avg_profiles.append(np.average(profsplits, axis=0, weights=self.split_wgts))
        if self.avg_img is None:
            self.avg_img = np.average(self.img_splits, axis=0, weights=self.split_wgts)

        self.r = np.arange(1, self.avg_img.shape[0]//2) * rad_in_Mpc / (self.avg_img.shape[0]//2)  # unbinned radius variable in Mpc
        # if self.r not equal to profile_splits.shape[2], print warning
        if len(self.r) != self.avg_profiles[0].shape[0]:
            print("Warning: r and profile_splits are different lengths.")
        # Initialize optional attributes to None
        self.covmat_full     = []
        self.cormat_full     = []
        self.errors_full     = []
        self.covmat_binned   = []
        self.cormat_binned   = []
        self.errors_binned   = [] # errors on the binned profile
        

        
    def set_split_wgts(self, additional_weights=None):
        # optionally replace split_wgts
        # if additional_weights is None, weights depend only on number of peaks in each split
        if additional_weights is None:
            additional_weights = np.ones(self.Nsamples)
        self.split_wgts = self.Npks_splits / np.average(self.Npks_splits) * additional_weights
    def set_average_profiles(self): # Option to call this by hand, to reset the profile, if the weights have changed
        self.avg_profiles = []
        for m,profsplits in enumerate(self.profile_splits):
            self.avg_profiles.append(np.average(profsplits, axis=0, weights=self.split_wgts))
    def set_avg_profiles_binned(self, binsize):
        # a list of the average binned profiles for each multipole moment m. Length m_max, each element shape (n_bins,)
        self.avg_profiles_binned = []
        for m,avgprof in enumerate(self.avg_profiles):
            binned_prof, binned_r = bin_profile(self.r, avgprof, self.rad_in_Mpc, binsize)
            self.avg_profiles_binned.append(np.asarray(binned_prof))
        self.r_binned = np.asarray(binned_r) # set binned r as whatever the last binned r was. These should all be the same.

    def set_custom_bin_m_avg(self, m, custom_bins):
        # rebin the mth multipole moment of the profiles
        custom_profile_m = [np.average(self.avg_profiles[m][custom_bins[i]:custom_bins[i+1]]) for i,bin in enumerate(custom_bins[:-1])]
        self.avg_profiles_binned[m] = np.asarray(custom_profile_m)
    def set_profile_splits_binned(self, binsize): # bin the profile of each split
        if not self.__has_splits__:
            print("No splits to bin.")
            return
        else:
            profile_splits_binned = []
            for m,profsplits in enumerate(self.profile_splits):
                profile_splits_binned_m = []
                for split in profsplits:
                    binned_prof, binned_r = bin_profile(self.r, split, self.rad_in_Mpc, binsize)
                    profile_splits_binned_m.append(np.asarray(binned_prof))
                profile_splits_binned.append(np.asarray(profile_splits_binned_m))
            self.profile_splits_binned = profile_splits_binned # list with len(m_max), each element shape (n_splits, n_bins)
            # not making into array because each element may have different shape after reassignment; see set_custom_bin_m
    def set_custom_bin_m_splits(self, m, custom_bins):
        # rebin the mth multipole moment of the profiles
        if not self.__has_splits__:
            print("No splits to bin.")
            return
        else:
            custom_profile_m = [np.average(self.profile_splits[m][:,custom_bins[i]:custom_bins[i+1]], axis=1) for i,bin in enumerate(custom_bins[:-1])]
            self.profile_splits_binned[m] = np.asarray(custom_profile_m)
    def set_covariance_full(self):
        # set the covariance matrix for the full profile
        if not self.__has_splits__:
            print("No splits to bin.")
            return
        else:
            for m,profsplits in enumerate(self.profile_splits):
                covmat, cormat = covariances(profsplits, self.split_wgts, self.Nsamples)
                self.covmat_full.append(covmat)
                self.cormat_full.append(cormat)
                self.errors_full.append(np.sqrt(np.diag(covmat)))
    def set_covariance_binned(self):
        # set the covariance matrix for the binned profile
        if not self.__has_splits__:
            print("No splits to bin.")
            return
        else:
            for m, profsplits in enumerate(self.profile_splits_binned):
                covmat, cormat = covariances(profsplits, self.split_wgts, self.Nsamples)
                self.covmat_binned.append(covmat)
                self.cormat_binned.append(cormat)
                self.errors_binned.append(np.sqrt(np.diag(covmat)))
    def bin_and_get_stats(self, binsize):
        self.set_profile_splits_binned(binsize)
        self.set_avg_profiles_binned(binsize)
        self.set_covariance_full()
        self.set_covariance_binned()
        