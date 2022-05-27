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

def bin_profile(Cr_m, rad_in_mpc, binsize): # binsize in Mpc
    binsize = binsize * u.Mpc
    rad_in_pix = len(Cr_m)
    mpc_per_pix = rad_in_mpc / rad_in_pix
    print(mpc_per_pix)
    pix_per_bin = int(round((binsize / mpc_per_pix).value))
    print(pix_per_bin)
    binned_prof = []
    for i in range(0,rad_in_pix,pix_per_bin):
        binned_prof.append(np.mean(Cr_m[i:i+pix_per_bin]))
    r = [i + 0.5 for i in range(len(binned_prof))]
    step_in_mpc = rad_in_mpc/len(r)
    return binned_prof, r*step_in_mpc

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
    y_mean        = pkl['prof_allreg'][:,m]
    y_mean_binned = pkl['binnedprof_allreg'][:,m]
    binned_r      = pkl['binned_r']
    npks_reg      = np.asarray([pkl['npks_reg{:d}'.format(reg)] for reg in range(nreg)])
    npks_tot      = pkl['npks_total']
    ys_all_regions_full = []
    ys_all_regions_binned = []
    for reg in range(nreg):
        y_binned = pkl['binnedprof_reg{:d}'.format(reg)][:,m]
        y = pkl['profs_reg{:d}'.format(reg)][:,m]
        ys_all_regions_binned.append(y_binned)
        ys_all_regions_full.append(y)
    weights = npks_reg / np.average(npks_reg)
    covmat_binned, cormat_binned = covariances(ys_all_regions_binned, weights, nreg)
    covmat_full, cormat_full = covariances(ys_all_regions_full, weights, nreg)
    errors_binned = np.sqrt(np.diag(covmat_binned))
    errors_full = np.sqrt(np.diag(covmat_full))
    return(np.arange(len(y_mean)), y_mean, covmat_full, cormat_full, errors_full, binned_r, y_mean_binned, covmat_binned, cormat_binned, errors_binned, npks_tot)

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
        Cr = np.zeros((n, mmax))
        Sr = np.zeros((n, mmax))

        for i in range(0, n):
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
    
    rad_in_pix = base_shape[0]//2
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

def hankel_multi(hankel_list, npks_list, d_low, slice_width, arcmin_per_pix, nstack=None, skip_list=None, ngal=False):
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
                    resized     = prof_func(np.linspace(0,len(prof_to_add)-1,240))
                    if ngal:
                        hankel_array[:,m] += ngal_convert_prof(resized, mid.value, arcmin_per_pix)
                    else:
                        hankel_array[:,m]  += resized
            else:
                print("Skipping array {:d} in the stack, assuming there's missing data in {:d} to {:d} Mpc slice.\n".format(c, i, uplim))
        c += 1
    return hankel_array / sum(npks_list), rad_in_mpc

