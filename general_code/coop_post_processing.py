import os
from astropy.io import fits
import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from astropy.cosmology import Planck15 as cosmo, z_at_value
from PIL import Image
import math

def bin_profile(Cr_m, rad_in_mpc, binsize): # binsize in Mpc
    binsize = binsize * u.Mpc
    rad_in_pix = len(Cr_m)
    mpc_per_pix = rad_in_mpc / rad_in_pix
    pix_per_bin = int(round((binsize / mpc_per_pix).value))
    binned_y = []
    for i in range(0,rad_in_pix,pix_per_bin):
        binned_y.append(np.mean(Cr_m[i:i+pix_per_bin]))
    r = [i + 0.5 for i in range(len(binned_y))]
    step_in_mpc = rad_in_mpc/len(r)
    return binned_y, r*step_in_mpc

def covariances(y_list,weights):
    y_array = np.asarray(y_list)
    return(np.cov(y_array.T, aweights=weights), np.corrcoef(y_array.T))

def get_img(stackfile, pkfile):
    stack = fits.open(stackfile)
    hdr = stack[0].header
    img = stack[0].data
    stack.close()
    pks    = fits.open(pkfile)
    npeaks = len(pks[0].data)
    pks.close()
    return hdr, img, npeaks


def ngal_convert(img, mid_slice_distance, arcmin_per_pix):
    z = z_at_value(cosmo.comoving_distance, mid_slice_distance * u.Mpc)
    mpc_per_arcmin = cosmo.kpc_comoving_per_arcmin(z).to(u.Mpc / u.arcmin)
    len_in_mpc = img.shape[0] * arcmin_per_pix * mpc_per_arcmin # the radius of the first image in comoving Mpc
    wid_in_mpc = img.shape[1] * arcmin_per_pix * mpc_per_arcmin # the radius of the first image in comoving Mpc
    sq_pix  = img.shape[0]*img.shape[1]
    sq_mpc  = len_in_mpc.value*wid_in_mpc.value
    print(sq_pix, sq_mpc)
    img = img * sq_pix/sq_mpc
    return(img)


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


def stack_multi(stack_list, npks_list, d_low, slice_width, arcmin_per_pix, nstack=None, skip_list=None):
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
                img_to_add = stack_list[c][xmin:xmax, xmin:xmax] * npks_list[c]
                pil_img    = Image.fromarray(img_to_add)
                pil_img_rs = pil_img.resize(base_shape)
                resized    = np.array(pil_img_rs.getdata()).reshape(pil_img_rs.size)
                im_array  += resized
            else:
                print("Skipping array {:d} in the stack, assuming there's missing data in {:d} to {:d} Mpc slice.\n".format(c, i, uplim))
        c += 1
    return im_array / sum(npks_list), rad_in_mpc

def hankel_multi(hankel_list, npks_list, d_low, slice_width, arcmin_per_pix, nstack=None, skip_list=None):
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
                    hankel_array[:,m]  += resized
            else:
                print("Skipping array {:d} in the stack, assuming there's missing data in {:d} to {:d} Mpc slice.\n".format(c, i, uplim))
        c += 1
    return hankel_array / sum(npks_list), rad_in_mpc

