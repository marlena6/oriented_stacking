import numpy as np
import coop_setup_funcs as csf
from astropy.cosmology import Planck18 as cosmo, z_at_value
import astropy.units as u
from scipy.integrate import quad
from scipy.stats import norm
import matplotlib.pyplot as plt
import healpy as hp
from astropy.io import fits
import math

pofz_weight = True
nside = 4096
mode  = 'buzzard'        
sigma_z_1pz = .05 #sigma_z/(1+z)

if mode=='buzzard':
    mask = hp.read_map("/mnt/raid-cita/mlokken/data/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_hpx_4096.fits")
elif mode=='websky':
    mask = None
def integrate_normal(photoz, sigma_z, low, hi):
    import time
    intgl = []
    times = []
    size  = len(photoz)
    c = 0
    for pz in photoz:
        start = time.time()
        intgl.append((quad(lambda x: norm.pdf(x, pz, sigma_z[c]), low, hi, limit=5))[0]) # only return the integral value, element 0 of a tuple
        end = time.time()
        times.append(end-start)
        start2 = time.time()
        c+=1
        if c%1000==0:
            print("{:0.1f}% elapsed".format(c/size*100))
        if c==10000:
            break
    
    print("Time of integration per galaxy:", np.average(times))
    return np.asarray(intgl)

def modify_pdist(photoz, sigma_z):
    import time
    manygals = []
    times = []
    m = 0
    for pz in photoz:
        start = time.time()
        mod = gauss_x*sigma_z[m]+pz # stretch the distribution by sigma_z, the original was sigma = 1
        manygals.extend(mod)
        print(mod)
        plt.plot(mod, np.full(len(mod), 1), 'o')
        plt.plot([pz-0.056374516/2., pz-0.056374516/2.], [-1,3], 'r', linestyle='--') 
        plt.plot([pz+0.056374516/2., pz+0.056374516/2.], [-1,3], 'r', linestyle='--')
        plt.plot([pz-.15/2., pz-.15/2.], [-1,3], 'k', linestyle='solid')
        plt.plot([pz+0.15/2., pz+0.15/2.], [-1,3], 'k', linestyle='solid')
        plt.show()
        plt.clf()
        asdf
        end = time.time()
        times.append(end-start)
        if c%1000==0:
            print("{:0.1f}% elapsed".format(c/size*100))
    print("Time per galaxy:", np.average(times))
    return np.asarray(manygals)

    
# import catalog
if mode=='websky':
    pzcat = np.load("/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/mock_maglim/mock_maglim_photoz.npy")
    ra, dec, gal_photoz = pzcat[:,0], pzcat[:,1], pzcat[:,2]
elif mode=='buzzard':
    with fits.open("/mnt/raid-cita/mlokken/buzzard/catalogs/maglim_buzz_Ndensity_1.fits") as buzzard:
        hdr = buzzard[1].header
        dat = buzzard[1].data
    ra, dec = dat['ra'], dat['dec']
    gal_photoz = dat['DNF_ZMEAN']
    sigma_z = dat['DNF_ZMEAN']-dat['DNF_ZMC'] # according to Shivam, a decent estimate of the sigma_z

size = len(ra)
print(size, "galaxies.")
if mode=='websky':
    sigma_z = sigma_z_1pz*(1+gal_photoz)


minz = 0.3
maxz = 0.6
# define the bins
dlist     = csf.dlist(minz=minz, maxz=maxz, slice_width=200)
dlist_off = csf.dlist(minz=minz, maxz=maxz, slice_width=200, offset=100)
if pofz_weight:    
    # make the standard Gaussian distribution sampling
    gauss_x = []
    fact    = 13
    CDF_sampling = np.linspace(0.5,1,math.ceil(fact/2))
    finex = np.linspace(-5,5,200)
    CDF_options  = norm.cdf(finex) # array
    gauss_x  = np.zeros(len(CDF_sampling))
    m = 0
    for c in CDF_sampling:
        diff = abs(CDF_options - np.full(len(CDF_options),c))
        gauss_x[m]=finex[np.where(diff ==min(diff))]
        m+=1

    gauss_x = np.concatenate((-1*gauss_x[1:],gauss_x))
    #plt.plot(gauss_x,np.full(len(gauss_x), 1), 'o')
    #plt.show()
    #plt.clf()

    photoz_mult = modify_pdist(gal_photoz, sigma_z)
    print(len(gal_photoz), len(photoz_mult))
    ra_mult  = np.zeros(len(photoz_mult))
    dec_mult = np.zeros(len(photoz_mult))
    for m in range(len(gal_photoz)):
        ra_mult[m*fact:m*fact+fact]=ra[m]
        dec_mult[m*fact:m*fact+fact]=dec[m]
    for i in range(len(dlist)):
        dbin = dlist[i]
        binmin,binmax = dbin[0], dbin[1]
        binmin_z, binmax_z = z_at_value(cosmo.comoving_distance, binmin*u.Mpc), z_at_value(cosmo.comoving_distance, binmax*u.Mpc)
        center_z = z_at_value(cosmo.comoving_distance, (binmin+binmax)/2.*u.Mpc)
        binmin_arr = np.full(size, binmin_z.value)
        binmax_arr = np.full(size, binmax_z.value)
        print("Bin ranging from {:d} to {:d} Mpc, {:.3f} to {:.3f} in redshift.".format(binmin,binmax,binmin_z.value, binmax_z.value))
        thisbin = (photoz_mult > binmin_z) & (photoz_mult < binmax_z)
        Nbin = np.sum(thisbin)
        print("{:d} galaxies in bin.".format(Nbin))
        z_in_bin  = photoz_mult[thisbin]
        thetaphi = np.zeros((len(ra_mult[thisbin]),2))
        theta,phi = csf.DeclRatoThetaPhi(dec_mult[thisbin], ra_mult[thisbin])
        thetaphi[:,0]=theta
        thetaphi[:,1]=phi
        # now make the map
        print(len(theta))
        print(len(phi))
        weight  = np.full(len(theta), 1/fact)
        odmap = csf.get_od_map(nside, theta, phi, mask, 0, wgt=weight)
        if mode=='websky':
            outfile = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/mock_maglim/photoz_weighted/odmap_mock_maglim_photoz_wgt_distMpc{:d}_{:d}_0arcmin.fits".format(binmin, binmax)
        elif mode=='buzzard':
            outfile = "/mnt/raid-cita/mlokken/buzzard/testing_photoz/photoz_weighted/odmap_mock_maglim_photoz_mult_distMpc{:d}_{:d}_0arcmin.fits".format(binmin, binmax)
        print("Writing map to %s" %outfile)
        hp.write_map(outfile, odmap, overwrite=True)
    for i in range(len(dlist_off)):
        dbin = dlist_off[i]
        binmin,binmax = dbin[0], dbin[1]
        binmin_z, binmax_z = z_at_value(cosmo.comoving_distance, binmin*u.Mpc), z_at_value(cosmo.comoving_distance, binmax*u.Mpc)
        center_z = z_at_value(cosmo.comoving_distance, (binmin+binmax)/2.*u.Mpc)
        binmin_arr = np.full(size, binmin_z.value)
        binmax_arr = np.full(size, binmax_z.value)
        print("Bin ranging from {:d} to {:d} Mpc, {:.3f} to {:.3f} in redshift.".format(binmin,binmax,binmin_z.value, binmax_z.value))
        thisbin = (photoz_mult > binmin_z) & (photoz_mult < binmax_z)
        Nbin = np.sum(thisbin)
        print("{:d} galaxies in bin.".format(Nbin))
        z_in_bin  = photoz_mult[thisbin]
        thetaphi = np.zeros((len(ra_mult[thisbin]),2))
        theta,phi = csf.DeclRatoThetaPhi(dec_mult[thisbin], ra_mult[thisbin])
        thetaphi[:,0]=theta
        thetaphi[:,1]=phi
        # now make the map
        print(len(theta))
        print(len(phi))
        weight  = np.full(len(theta), 1/fact)
        odmap = csf.get_od_map(nside, theta, phi, mask, 0, wgt=weight)
        if mode=='websky':
            outfile = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/mock_maglim/photoz_weighted/odmap_mock_maglim_photoz_wgt_distMpc{:d}_{:d}_0arcmin.fits".format(binmin, binmax)
        elif mode=='buzzard':
            outfile = "/mnt/raid-cita/mlokken/buzzard/testing_photoz/photoz_weighted/odmap_mock_maglim_photoz_mult_distMpc{:d}_{:d}_0arcmin.fits".format(binmin, binmax)
        print("Writing map to %s" %outfile)
        hp.write_map(outfile, odmap, overwrite=True)

else:
    # unweighted
    thetaphi, dists = csf.radec_to_thetaphi_sliced(ra, dec, gal_photoz, minz, maxz, 200)
    for i in range(len(dlist)):
        dbin = dlist[i]
        binmin,binmax = dbin[0], dbin[1]
        print("Bin ranging from {:d} to {:d} Mpc.".format(binmin,binmax))
        if len(thetaphi[i])!=0:
            odmap = csf.get_od_map(nside, thetaphi[i][:,0], thetaphi[i][:,1], mask, 0)
            if mode=='websky':
                outfile = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/mock_maglim/photoz_unweighted/odmap_mock_maglim_photoz_nowgt_distMpc{:d}_{:d}_0arcmin.fits".format(binmin, binmax)
            elif mode=='buzzard':
                outfile = "/mnt/raid-cita/mlokken/buzzard/testing_photoz/photoz_unweighted/odmap_mock_maglim_photoz_nowgt_distMpc{:d}_{:d}_0arcmin.fits".format(binmin, binmax)
            print("Writing map to %s" %outfile)
            hp.write_map(outfile, odmap, overwrite=True)

    thetaphi_off, dists_off = csf.radec_to_thetaphi_sliced(ra, dec, gal_photoz, minz, maxz, 200, offset=100)
    for i in range(len(dlist_off)):
        dbin = dlist_off[i]
        binmin,binmax = dbin[0], dbin[1]
        print("Bin ranging from {:d} to {:d} Mpc.".format(binmin,binmax))
        if len(thetaphi[i])!=0:
            odmap = csf.get_od_map(nside, thetaphi_off[i][:,0], thetaphi_off[i][:,1], mask, 0)
            if mode=='websky':
                outfile = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/mock_maglim/photoz_unweighted/odmap_mock_maglim_photoz_nowgt_distMpc{:d}_{:d}_0arcmin.fits".format(binmin, binmax)
            elif mode=='buzzard':
                outfile = "/mnt/raid-cita/mlokken/buzzard/testing_photoz/photoz_unweighted/odmap_mock_maglim_photoz_nowgt_distMpc{:d}_{:d}_0arcmin.fits".format(binmin, binmax)
            print("Writing map to %s" %outfile)
            hp.write_map(outfile, odmap, overwrite=True)
 
el
