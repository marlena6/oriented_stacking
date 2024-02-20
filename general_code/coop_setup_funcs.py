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

def get_od_map(nside, theta, phi, mask=None, smth=0, wgt=1, beam='gaussian'): # smoothing scale in arcsec
    import healpy as hp
    map   = np.zeros((hp.nside2npix(nside)))
    pix = hp.ang2pix(nside,theta,phi)
    np.add.at(map, pix, wgt)
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
        if beam=='gaussian':
            smthmap = hp.sphtfunc.smoothing(newmap, fwhm = np.deg2rad(smth/3600.), pol=False)
        elif beam=='tophat':
            smthmap = hp.sphtfunc.smoothing(newmap, fwhm = np.deg2rad(smth/3600.), pol=False, beam_window=tophat_beam(smth/3600.))
    else:
        smthmap = newmap
    return smthmap

def get_nd_map(nside, theta, phi, mask, smth=0, wgt=1, beam='gaussian'): # smoothing scale in arcsec
    import healpy as hp
    map   = np.zeros((hp.nside2npix(nside)))
    pix = hp.ang2pix(nside,theta,phi)
    np.add.at(map, pix, wgt)
    if mask is not None:
        map = map * mask
    if smth != 0:
        if beam=='gaussian':
            smthmap = hp.sphtfunc.smoothing(map, fwhm = np.deg2rad(smth/3600.), pol=False)
        elif beam=='tophat':
            smthmap = hp.sphtfunc.smoothing(map, fwhm = np.deg2rad(smth/3600.), pol=False, beam_window=tophat_beam(smth/3600.))
    else:
        smthmap = map
    return smthmap

def get_nu_map(nside, theta, phi, mask, smth=0, wgt=1): # smoothing scale in arcsec
    import healpy as hp
    map = np.zeros((hp.nside2npix(nside)))
    pix = hp.ang2pix(nside,theta,phi)
    np.add.at(map, pix, wgt)
    if mask is not None:
        map = map * mask
        npix = sum(mask)
    else:
        npix = hp.nside2npix(nside)
    mean = sum(map)/npix
    newmap   = map/mean - 1
    if smth != 0:
        smthmap = hp.sphtfunc.smoothing(newmap, fwhm = np.deg2rad(smth/3600.), pol=False)*mask
    else:
        smthmap = newmap*mask
    mean_od = sum(smthmap)/sum(mask)
    rms     = np.sqrt(sum((smthmap-mean_od)**2*mask)/sum(mask))
    print("rms = ", rms)
    print("mean of smoothed overdensity map = ", mean_od)
    nu = ((smthmap-mean_od)*mask) / rms
    return nu
    
def get_radecz(filepath, min_mass=None, max_mass=None, return_mass=False, return_id=False):
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
        if 'redmapper' in filepath:
            id = dat['mem_match_id']
        else:
            id = dat['id']
    elif ".npy" in filepath:
        print("Peak Patch catalog entered.")
        halos = np.load(filepath)
        M     = halos[:,0]
        id = np.arange(len(halos)) # Websky has no special ID, simply label clusters at this point

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
    to_return = [ra,dec,z]
    if return_mass:
        to_return.append(M)
    if return_id:
        to_return.append(id)
    return to_return

def get_radeczlambda(filepath, min_mass=None, return_mass=False, return_id=False):
    # load catalog and get ra, dec
    if "redmapper" in filepath or "redmagic" in filepath:
        hdu = fits.open(filepath)
        dat = hdu[1].data
        hdr = hdu[1].header
        hdu.close()
        ra  = dat['RA']
        dec = dat['DEC']
        z = dat['Z_LAMBDA']
        if ('buzzard' in filepath) or ('cardinal' in filepath):
            richness = dat['lambda']
        else:
            richness = dat['lambda_chisq']
        if 'redmapper' in filepath:
            id = dat['mem_match_id']
        else:
            id = dat['id']
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
        id = np.arange(len(halos)) # Websky has no special ID, simply label clusters at this point
    else:
        print("unrecognized file format")
    to_return = [ra,dec,z,richness]
    if return_mass:
        to_return.append(M)
    if return_id:
        to_return.append(id)
    return to_return

def get_nu(od_filename, ra, dec, mask=None):
    print("faster nu")
    import healpy as hp
    ''' returns the nu values at the positions of the input ra, dec'''
    od_map = read_amp_map(od_filename)
    nside  = hp.get_nside(od_map)
    pixels = hp.ang2pix(nside, ra, dec, lonlat=True)
    if mask is not None:
        mean_od = np.sum(od_map*mask)/np.sum(mask)
        rms = np.sqrt(np.sum((od_map-mean_od)**2*mask)/np.sum(mask))
    else:
        mean_od = np.sum(od_map)/od_map.size
        rms = np.sqrt(np.sum((od_map-mean_od)**2)/od_map.size)
    print("rms = ", rms)
    print("mean of overdensity map = ", mean_od)
    if mask is not None:
        nu = ((od_map-mean_od)*mask) / rms
    else:
        nu = (od_map-mean_od)/rms
    hp.mollview(nu)
    return(nu[pixels])

def get_x_e(e_filename, ra, dec, mask=None):
    import healpy as hp
    ecc_map0 = hp.read_map(e_filename, field=0)
    ecc_map1 = hp.read_map(e_filename, field=1)    
    ecc_map2 = hp.read_map(e_filename, field=2)
    nside    = hp.get_nside(ecc_map0)
    pixels   = hp.ang2pix(nside, ra, dec, lonlat=True)
    e  = np.sqrt((ecc_map1**2 + ecc_map2**2)/(ecc_map0**2))
    #ecc_map0 = del^2(F)
    if mask is None:
        mean_del2 = np.sum(ecc_map0)/ecc_map0.size
        rms_2 = np.sqrt(np.sum((ecc_map0-mean_del2)**2)/ecc_map0.size)
        x = ecc_map0 / rms_2
    else:
        mean_del2 = np.sum(ecc_map0*mask)/np.sum(mask)
        rms_2 = np.sqrt(np.sum((ecc_map0-mean_del2)**2*mask)/np.sum(mask))
        x = ecc_map0*mask / rms_2
    print("mean of del^2(F) = ", mean_del2)
    print("rms of del^2(F)  = ", rms_2)
    print("Map of x = del^2(F)")
    return e[pixels], x[pixels]

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

def radec_to_thetaphi_sliced(ra, dec, z_arr, minz=None, maxz=None, slice_width=None, min_lambda=None, max_lambda=None, richness=None, tag=None, offset=None, zbins=None):

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
    dlist    = []
    idx_list = []
    if zbins is not None:
        for z in zbins:
            # limit my sample to objects which exist in this redshift bin
            z_slice_min,z_slice_max = z[0], z[1]
            in_bin = np.where(np.logical_and(z_arr>z_slice_min,z_arr<z_slice_max))
            ra_in    = ra[in_bin]
            dec_in   = dec[in_bin]
            z_in     = z_arr[in_bin]
            idx_list.append(in_bin)
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
            idx_list.append(in_bin)
            ra_in  = ra[in_bin]
            dec_in   = dec[in_bin]
            z_in     = z_arr[in_bin]
            print("Found %d objects in the distance slice between %d and %d Mpc.\n" %(len(ra_in), int(dist_slice_min.value), int(dist_slice_max.value)))
            thetaphi = np.zeros((len(ra_in),2))
            theta,phi = DeclRatoThetaPhi(dec_in, ra_in)
            thetaphi[:,0]=theta
            thetaphi[:,1]=phi
            thetaphi_list.append(thetaphi)
            dlist.append([int(dist_slice_min.value), int(dist_slice_max.value)])
    return(thetaphi_list, dlist, idx_list)
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

def read_amp_map(amp_file):
    import healpy as hp
    ''' reads in an amplitude map and returns it '''
    # if reading with healpy throws an error, rewrite the header to have the correct TTYPE3
    if fits.open(amp_file)[1].header['TTYPE3'] != 'ID2':    
        amp_map = fits.open(amp_file)
        amp_map[1].header['TTYPE3'] = 'ID2'
        amp_map.writeto(amp_file, overwrite=True)
        amp_map.close()

    return hp.read_map(amp_file, field=0)

def tophat_beam(scale, lmax=8000):
    import healpy as hp
    import numpy as np
    ''' takes beam size in degrees, outputs tophat beam '''
    theta = np.linspace(0, np.deg2rad(scale))
    beam  = np.ones(len(theta))
    beam = hp.beam2bl(beam, theta, lmax)
    return beam

def tophat_smooth_pixell(imap, scale, lmax=30000):
    ''' takes pixell map, tophat beam size in degrees '''
    beam = tophat_beam(scale)
    imap_filt = curvedsky.filter(imap, beam, lmax=lmax)
    return imap_filt