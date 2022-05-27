import numpy as np
from astropy.io import fits
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u
import os
import sys

def DeclRatoThetaPhi(decl,RA):
    return np.radians(-decl+90.),np.radians(RA)

def ThetaPhitoRaDec(theta,phi):
    return np.rad2deg(phi), -1*(np.rad2deg(theta)-90.)

def get_od_map(nside, theta, phi, mask, smth, mass=None): # smoothing scale in arcsec
    import healpy as hp
    map   = np.zeros((hp.nside2npix(nside)))
    pix = hp.ang2pix(nside,theta,phi)
    if mass is None:
        weight = 1.
    else:
        weight = mass/(10**12)
    np.add.at(map, pix, weight)
    if mask is not None:
        map = map * mask
        npix = sum(mask)
    else:
        npix = hp.nside2npix(nside)
    mean = sum(map)/npix
    newmap   = map/mean - 1
    if mask is not None:
        newmap = newmap * mask
    print("Mean of number density map: ", mean)
    print("Mean of overdensity map: ", sum(newmap)/npix)
    if smth != 0:
        smthmap = hp.sphtfunc.smoothing(newmap, fwhm = np.deg2rad(smth/3600.), pol=False)
    else:
        smthmap = newmap
    return smthmap

def get_nd_map(nside, theta, phi, mask, smth, mass=None): # smoothing scale in arcsec
    import healpy as hp
    map   = np.zeros((hp.nside2npix(nside)))
    pix = hp.ang2pix(nside,theta,phi)
    if mass is None:
        weight = 1.
    else:
        weight = mass/(10**12)
    np.add.at(map, pix, weight)
    if mask is not None:
        map = map * mask
    if smth != 0:
        smthmap = hp.sphtfunc.smoothing(map, fwhm = np.deg2rad(smth/3600.), pol=False)
    else:
        smthmap = map
    return smthmap

def get_radecz(filepath, min_mass=None, max_mass=None, return_mass=False):
    # load catalog and get ra, dec                                                                              
    if "redmapper" in filepath or "redmagic" in filepath:
        print("Redmapper or redmagic catalog entered.")
        hdu = fits.open(filepath)
        dat = hdu[1].data
        hdr = hdu[1].header
        hdu.close()
        ra  = dat['RA']
        dec = dat['DEC']
        z   = dat['Z_LAMBDA']
    elif ".npy" in filepath:
        print("Peak Patch catalog entered.")
        halos = np.load(filepath)
        M     = halos[:,0]
        if (min_mass is not None) or (max_mass is not None): # if there is either a minimum or maximum mass
            if min_mass is not None and max_mass is None:
                include = M > min_mass
            elif min_mass is not None and max_mass is not None:
                include = (M > min_mass) & (M < max_mass)
            elif min_mass is None and max_mass is not None:
                include = M < max_mass
            M   = M[include]
            ra  = halos[:,4][include]
            dec = halos[:,5][include]
            z   = halos[:,6][include]
        else:
            ra  = halos[:,4]
            dec = halos[:,5]
            z   = halos[:,6]
    else:
        print("unrecognized file format")
    if return_mass:
        return(ra,dec,z, M)
    else:
        return(ra,dec,z)

def get_radeczlambda(filepath, min_mass=None, return_mass=False):
    # load catalog and get ra, dec
    if "redmapper" in filepath or "redmagic" in filepath:
        hdu = fits.open(filepath)
        dat = hdu[1].data
        hdr = hdu[1].header
        hdu.close()
        ra  = dat['RA']
        dec = dat['DEC']
        z = dat['Z_LAMBDA']
        if 'buzzard' in filepath:
            richness = dat['lambda']
        else:
            richness = dat['LAMBDA_CHISQ']
    elif ".npy" in filepath:
        # takes npy file with format [M,x,y,z,ra,dec,redshift]
        halos = np.load(filepath)
        M     = halos[:,0]
        if min_mass is not None:
            include = M > min_mass
            M   = M[include]
            ra  = halos[:,4][include]
            dec = halos[:,5][include]
            z   = halos[:,6][include]
        else:
            ra  = halos[:,4]
            dec = halos[:,5]
            z   = halos[:,6]
        richness = mass_to_richness(M,z)
    else:
        print("unrecognized file format")
    if return_mass:
        return(ra,dec,z,richness,M)
    else:
        return(ra,dec,z,richness)

def mass_to_richness(M, z):
    # M (mass) can either be a single value or array                                                                                                
    # implements the richness-mass relation from McClintock et al 2018
    lambda_0 = 40.
    z_0 = 0.35
    F_lambda = 1.356
    G_z = -0.3
    M_0 =  3.081 * 10**14 # in units of M_sun
    richness = lambda_0*(((M/M_0)*(((1+z)/(1+z_0))**-G_z))**(1/F_lambda))
    return(richness)

def richness_to_mass(richness, z):
    # richness can either be a single value or array
    # implements the richness-mass relation from McClintock et al 2018
    lambda_0 = 40.
    z_0 = 0.35
    F_lambda = 1.356
    G_z = -0.3
    M_0 =  3.081 * 10**14 # in units of M_sun
    mass = M_0*(richness/lambda_0)**F_lambda*((1+z)/(1+z_0))**G_z    
    return(mass)

def dlist(minz=None, maxz=None, slice_width=None, offset=0, zbins=None):
    dlist = []
    if zbins is None and minz is None:
        sys.exit("Need to input one of either zbins, minz.")
    if zbins is not None:
        for z in zbins:
            # limit my sample to objects which exist in this redshift bin
            dist_slice_min, dist_slice_max = cosmo.comoving_distance(z[0]),cosmo.comoving_distance(z[1])
            dlist.append([int(dist_slice_min.value), int(dist_slice_max.value)])    
    else:
        nbins = int((cosmo.comoving_distance(maxz).value - cosmo.comoving_distance(minz).value) // slice_width)
        print("Number of distance bins: %d" %nbins)
        for i in range(nbins):
            dist_slice_min = cosmo.comoving_distance(minz)+float(offset)*u.Mpc + slice_width*u.megaparsec*i
            dist_slice_max = dist_slice_min + slice_width*u.megaparsec
            dlist.append([int(dist_slice_min.value), int(dist_slice_max.value)])
    return dlist

def radec_to_thetaphi_sliced(ra, dec, z_arr, minz=None, maxz=None, slice_width=None, masses=None, min_lambda=None, max_lambda=None, richness=None, tag=None, offset=None, zbins=None):

## takes in data of ra, dec, z for objects and splits it up into slices in redshift.
## ra, dec, z_arr must all have same length and correspond to same objects.
## minz and maxz are the closest edge of the first slice and the furthest out you want to go (objects beyond this won't be included)
## outpath is the path of the output lists of (theta,phi) and full information about the objects
## min_lambda, max_lambda are the richness bounds for clusters
## richness is the array of cluster richnesses
## tag will be added after 'thetaphi' in each output filename

    # Go through distance bins starting with minimum cluster redshift (in DESxACT reg, this is 0.1)
    if slice_width==None and zbins==None:
        sys.exit("radec_to_thetaphi_sliced function error: Must enter zbins list or (minz, maxz, slice_width)")
    if not offset:
        offset = 0.
    thetaphi_list = []
    dlist = []
    Mlist = []
    if zbins is not None:
        for z in zbins:
            # limit my sample to objects which exist in this redshift bin
            z_slice_min,z_slice_max = z[0], z[1]
            in_bin = np.where(np.logical_and(z_arr>z_slice_min,z_arr<z_slice_max))
            ra_in    = ra[in_bin]
            dec_in   = dec[in_bin]
            z_in     = z_arr[in_bin]
            if masses is not None:
                M    = masses[in_bin]
                Mlist.append(M)
            print("Found %d objects in the distance slice between %.2f and %.2f.\n" %(len(ra_in), z_slice_min, z_slice_max))
            thetaphi = np.zeros((len(ra_in),2))
            theta,phi = DeclRatoThetaPhi(dec_in, ra_in)
            thetaphi[:,0]=theta
            thetaphi[:,1]=phi
            thetaphi_list.append(thetaphi)
            dist_slice_min, dist_slice_max = cosmo.comoving_distance(z[0]),cosmo.comoving_distance(z[1])
            dlist.append([int(dist_slice_min.value), int(dist_slice_max.value)])
    else:
        nbins = int((cosmo.comoving_distance(maxz).value - cosmo.comoving_distance(minz).value) // slice_width)
        print("Number of distance bins: %d" %nbins)
        for i in range(nbins):
            dist_slice_min = cosmo.comoving_distance(minz)+float(offset)*u.Mpc + slice_width*u.megaparsec*i
            dist_slice_max = dist_slice_min + slice_width*u.megaparsec
            z_slice_min    = z_at_value(cosmo.comoving_distance, dist_slice_min)
            z_slice_max    = z_at_value(cosmo.comoving_distance, dist_slice_max)
            # limit my sample to objects which exist in this redshift bin
            in_bin = np.where(np.logical_and(z_arr>z_slice_min,z_arr<z_slice_max))
            ra_in  = ra[in_bin]
            dec_in   = dec[in_bin]
            z_in     = z_arr[in_bin]
            if masses is not None:
                M    = masses[in_bin]
                Mlist.append(M)
            print("Found %d objects in the distance slice between %d and %d Mpc.\n" %(len(ra_in), int(dist_slice_min.value), int(dist_slice_max.value)))
            thetaphi = np.zeros((len(ra_in),2))
            theta,phi = DeclRatoThetaPhi(dec_in, ra_in)
            thetaphi[:,0]=theta
            thetaphi[:,1]=phi
            thetaphi_list.append(thetaphi)
            dlist.append([int(dist_slice_min.value), int(dist_slice_max.value)])
    if masses is None:
        return(thetaphi_list, dlist)
    else:
        return(thetaphi_list, dlist, Mlist)
'''
        # make an array called save_info with [lambda, ra, dec, theta, phi] or just [ra,dec,theta,phi]
        if richness is not None:
            save_info = np.zeros((len(ra_in), 6))
            richness_in = richness[in_bin_1]
            save_info[:,0] = richness_in
            save_info[:,1] = z_in
            save_info[:,2] = ra_in
            save_info[:,3] = dec_in
            save_info[:,4] = theta
            save_info[:,5] = phi
            if not tag:
                thetaphi_fn = "thetaphi_lambda_%d_to_%d_distMpc_%d_%d.txt" %(int(min_lambda),int(max_lambda),int(dist_slice_min.value),int(dist_slice_max.value))
                fullinfo_fn = "fullinfo_lambda_%d_to_%d_distMpc_%d_%d.txt" %(int(min_lambda),int(max_lambda),int(dist_slice_min.value),int(dist_slice_max.value))
            else:
                thetaphi_fn = "thetaphi_%s_lambda_%d_to_%d_distMpc_%d_%d.txt" %(tag, int(min_lambda),int(max_lambda),int(dist_slice_min.value),int(dist_slice_max.value))
                fullinfo_fn = "fullinfo_%s_lambda_%d_to_%d_distMpc_%d_%d.txt" %(tag, int(min_lambda),int(max_lambda),int(dist_slice_min.value),int(dist_slice_max.value))
                
        else:
            save_info = np.zeros((len(ra_in), 5))
            save_info[:,0] = z_in
            save_info[:,1] = ra_in
            save_info[:,2] = dec_in
            save_info[:,3] = theta
            save_info[:,4] = phi
            if not tag:
                thetaphi_fn = "thetaphi_distMpc_%d_%d.txt" %(int(dist_slice_min.value),int(dist_slice_max.value))
                fullinfo_fn = "fullinfo_distMpc_%d_%d.txt" %(int(min_lambda),int(max_lambda),int(dist_slice_min.value),int(dist_slice_max.value))
            else:
                thetaphi_fn = "thetaphi_%s_distMpc_%d_%d.txt" %(tag, int(dist_slice_min.value),int(dist_slice_max.value))
                fullinfo_fn = "fullinfo_%s_distMpc_%d_%d.txt" %(tag, int(min_lambda),int(max_lambda),int(dist_slice_min.value),int(dist_slice_max.value))
            
        saveas_tp   = os.path.join(outpath,thetaphi_fn)
        saveas_full = os.path.join(outpath,fullinfo_fn)
        np.savetxt(saveas_tp, thetaphi)
        np.savetxt(saveas_full, save_info)
        print("Theta phi file saved in %s, full info file saved in %s \n"%(saveas_tp, saveas_full))
'''
