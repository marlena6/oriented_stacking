import numpy as np
import coop_setup_funcs as csf
from astropy.cosmology import Planck18 as cosmo, z_at_value
import astropy.units as u
from scipy.integrate import quad
from scipy.stats import norm
import matplotlib.pyplot as plt
import healpy as hp

pofz_weight = False
nside = 4096
        
sigma_z_1pz = .05 #sigma_z/(1+z)

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

# import catalog
if mode=='websky':
    pzcat = np.load("/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/mock_maglim/mock_maglim_photoz.npy")
    ra, dec, gal_photoz = pzcat[:,0], pzcat[:,1], pzcat[:,2]
elif mode=='buzzard':
    with fits.open("/mnt/raid-cita/mlokken/buzzard/catalogs/maglim_buzz_Ndensity_1.fits") as buzzard:
        hdr = buzzard[1].header
        dat = buzzard[1].data
    ra, dec = dat['ra'], dat['dec']
    gal_photoz = dat['Z_MEAN']
    sigma_z = dat['Z_MEAN']-dat['Z_MC'] # according to Shivam, a decent estimate of the sigma_z
    
size = len(ra)
print(size, "galaxies.")
if mode=='websky':
    sigma_z = sigma_z_1pz*(1+gal_photoz)
extent  = 1.5
max_photoz = gal_photoz+extent*sigma_z
min_photoz = gal_photoz-extent*sigma_z
minz = 0.3
maxz = 0.6
# define the bins
dlist     = csf.dlist(minz=minz, maxz=maxz, slice_width=200)
dlist_off = csf.dlist(minz=minz, maxz=maxz, slice_width=200, offset=100)
if pofz_weight:
    for i in range(len(dlist)):
        dbin = dlist[i]
        binmin,binmax = dbin[0], dbin[1]
        binmin_z, binmax_z = z_at_value(cosmo.comoving_distance, binmin*u.Mpc), z_at_value(cosmo.comoving_distance, binmax*u.Mpc)
        center_z = z_at_value(cosmo.comoving_distance, (binmin+binmax)/2.*u.Mpc)
        binmin_arr = np.full(size, binmin_z.value)
        binmax_arr = np.full(size, binmax_z.value)
        print("Bin ranging from {:d} to {:d} Mpc, {:.3f} to {:.3f} in redshift.".format(binmin,binmax,binmin_z.value, binmax_z.value))
        # find the galaxies for which any part of the distribution lies within this bin
        # first find the ones that are 'completely' within this bin
        # all_in_bin        = (binmin_arr< min_photoz) & (binmax_arr > max_photoz)
        # print("{:d} galaxies completely within.".format(np.sum(all_in_bin))) -- there are none for this test
        # compute the integral once for galaxies close to center
        lower_edge_in_bin = (binmin_arr < min_photoz) & (min_photoz < binmax_arr) # the lower end of the p(z) dist is in the bin
        print("{:d} galaxies lower edge.".format(np.sum(lower_edge_in_bin)))
        upper_edge_in_bin = (binmin_arr < max_photoz) & (max_photoz < binmax_arr) # the upper end of the p(z) dist is in the bin
        print("{:d} galaxies upper edge.".format(np.sum(upper_edge_in_bin)))
        pofz_larger_bin   = (binmax_arr < max_photoz) & (min_photoz < binmin_arr) # the p(z) dist is wider than the bin
        print("{:d} galaxies wider than bin.".format(np.sum(pofz_larger_bin)))
        thisbin = lower_edge_in_bin | upper_edge_in_bin | pofz_larger_bin
        Nbin = np.sum(thisbin)
        print("{:d} galaxies in bin.".format(Nbin))
        # now assign weights for galaxies overlapping this bin
        pz_thisbin = gal_photoz[thisbin]
        w = np.zeros(len(pz_thisbin))
        center_w    = quad(lambda x: norm.pdf(x, center_z, sigma_z_1pz*(1+center_z)), binmin_z, binmax_z)[0]
        print(center_w, "z in bin weight")
        z_in_bin    = (pz_thisbin>binmin_z)&(pz_thisbin<binmax_z)
        print(np.sum(z_in_bin), "galaxies with central photoz in the bin")
        w[z_in_bin] = center_w # a lazy approximation
        print(np.sum((pz_thisbin+extent*sigma_z_1pz*(1+pz_thisbin))<np.full(Nbin, binmax_z.value)), "upper tail length")
        either_tail_end = (((pz_thisbin+extent*sigma_z_1pz*(1+pz_thisbin))<np.full(Nbin, binmax_z.value)) | ((pz_thisbin-(extent*sigma_z_1pz*(1+pz_thisbin)))>np.full(Nbin, binmin_z.value))) & (~z_in_bin)
        I1 = quad(lambda x: norm.pdf(x, binmin_z, sigma_z_1pz*(1+binmin_z)), binmin_z, binmax_z)[0]
        tailw = (I1+0.07)/2.
        print(tailw, "tail end in bin weight")
        w[either_tail_end] = tailw
        zeros = np.where(w==0)
        print(len(zeros), "this many zeros")
        #print("Calculating weights for all {:d} others.".format(len(w[~gal_near_center])))
        #w[~gal_near_center] = integrate_normal(pz_thisbin[~gal_near_center], sigma_z[thisbin][~gal_near_center], binmin_z, binmax_z)
        #w = integrate_normal(gal_photoz[thisbin], sigma_z, binmin_z, binmax_z)
        thetaphi = np.zeros((len(ra[thisbin]),2))
        theta,phi = csf.DeclRatoThetaPhi(dec[thisbin], ra[thisbin])
        thetaphi[:,0]=theta
        thetaphi[:,1]=phi
        # now make the map
        print(len(theta))
        print(len(phi))
        print(len(w))
        odmap = csf.get_od_map(nside, theta, phi, None, 0, wgt=w)
        outfile = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/mock_maglim/photoz_weighted/odmap_mock_maglim_photoz_wgt_distMpc{:d}_{:d}_0arcmin.fits".format(binmin, binmax)
        print("Writing map to %s" %outfile)
        hp.write_map(outfile, odmap, overwrite=False)
    for i in range(len(dlist_off)):
        dbin = dlist_off[i]
        binmin,binmax = dbin[0], dbin[1]
        binmin_z, binmax_z = z_at_value(cosmo.comoving_distance, binmin*u.Mpc), z_at_value(cosmo.comoving_distance, binmax*u.Mpc)
        center_z = z_at_value(cosmo.comoving_distance, (binmin+binmax)/2.*u.Mpc)
        binmin_arr = np.full(size, binmin_z.value)
        binmax_arr = np.full(size, binmax_z.value)
        print("Bin ranging from {:d} to {:d} Mpc, {:.3f} to {:.3f} in redshift.".format(binmin,binmax,binmin_z.value, binmax_z.value))
        # find the galaxies for which any part of the distribution lies within this bin
        # first find the ones that are 'completely' within this bin
        # all_in_bin        = (binmin_arr< min_photoz) & (binmax_arr > max_photoz)
        # print("{:d} galaxies completely within.".format(np.sum(all_in_bin))) -- there are none for this test
        # compute the integral once for galaxies close to center
        lower_edge_in_bin = (binmin_arr < min_photoz) & (min_photoz < binmax_arr) # the lower end of the p(z) dist is in the bin
        print("{:d} galaxies lower edge.".format(np.sum(lower_edge_in_bin)))
        upper_edge_in_bin = (binmin_arr < max_photoz) & (max_photoz < binmax_arr) # the upper end of the p(z) dist is in the bin
        print("{:d} galaxies upper edge.".format(np.sum(upper_edge_in_bin)))
        pofz_larger_bin   = (binmax_arr < max_photoz) & (min_photoz < binmin_arr) # the p(z) dist is wider than the bin
        print("{:d} galaxies wider than bin.".format(np.sum(pofz_larger_bin)))
        thisbin = lower_edge_in_bin | upper_edge_in_bin | pofz_larger_bin
        Nbin = np.sum(thisbin)
        print("{:d} galaxies in bin.".format(Nbin))
        # now assign weights for galaxies overlapping this bin
        pz_thisbin = gal_photoz[thisbin]
        w = np.zeros(len(pz_thisbin))
        center_w    = quad(lambda x: norm.pdf(x, center_z, sigma_z_1pz*(1+center_z)), binmin_z, binmax_z)[0]
        print(center_w, "z in bin weight")
        z_in_bin    = (pz_thisbin>binmin_z)&(pz_thisbin<binmax_z)
        print(np.sum(z_in_bin), "galaxies with central photoz in the bin")
        w[z_in_bin] = center_w # a lazy approximation
        print(np.sum((pz_thisbin+extent*sigma_z_1pz*(1+pz_thisbin))<np.full(Nbin, binmax_z.value)), "upper tail length")
        either_tail_end = (((pz_thisbin+extent*sigma_z_1pz*(1+pz_thisbin))<np.full(Nbin, binmax_z.value)) | ((pz_thisbin-(extent*sigma_z_1pz*(1+pz_thisbin)))>np.full(Nbin, binmin_z.value))) & (~z_in_bin)
        I1 = quad(lambda x: norm.pdf(x, binmin_z, sigma_z_1pz*(1+binmin_z)), binmin_z, binmax_z)[0]
        tailw = (I1+0.07)/2.
        print(tailw, "tail end in bin weight")
        w[either_tail_end] = tailw
        zeros = np.where(w==0)
        print(len(zeros), "this many zeros")
        #print("Calculating weights for all {:d} others.".format(len(w[~gal_near_center])))
        #w[~gal_near_center] = integrate_normal(pz_thisbin[~gal_near_center], sigma_z[thisbin][~gal_near_center], binmin_z, binmax_z)
        #w = integrate_normal(gal_photoz[thisbin], sigma_z, binmin_z, binmax_z)
        thetaphi = np.zeros((len(ra[thisbin]),2))
        theta,phi = csf.DeclRatoThetaPhi(dec[thisbin], ra[thisbin])
        thetaphi[:,0]=theta
        thetaphi[:,1]=phi
        # now make the map
        print(len(theta))
        print(len(phi))
        print(len(w))
        odmap = csf.get_od_map(nside, theta, phi, None, 0, wgt=w)
        outfile = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/mock_maglim/photoz_weighted/odmap_mock_maglim_photoz_wgt_distMpc{:d}_{:d}_0arcmin.fits".format(binmin, binmax)
        print("Writing map to %s" %outfile)
        hp.write_map(outfile, odmap, overwrite=False)

else:
    # unweighted
    thetaphi, dists = csf.radec_to_thetaphi_sliced(ra, dec, gal_photoz, minz, maxz, 200)
    for i in range(len(dlist)):
        dbin = dlist[i]
        binmin,binmax = dbin[0], dbin[1]
        print("Bin ranging from {:d} to {:d} Mpc.".format(binmin,binmax))
        odmap = csf.get_od_map(nside, thetaphi[i][:,0], thetaphi[i][:,1], None, 0)
        outfile = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/mock_maglim/photoz_unweighted/odmap_mock_maglim_photoz_nowgt_distMpc{:d}_{:d}_0arcmin.fits".format(binmin, binmax)
        print("Writing map to %s" %outfile)
        hp.write_map(outfile, odmap, overwrite=False)

    thetaphi_off, dists_off = csf.radec_to_thetaphi_sliced(ra, dec, gal_photoz, minz, maxz, 200, offset=100)
    for i in range(len(dlist_off)):
        dbin = dlist_off[i]
        binmin,binmax = dbin[0], dbin[1]
        print("Bin ranging from {:d} to {:d} Mpc.".format(binmin,binmax))
        odmap = csf.get_od_map(nside, thetaphi_off[i][:,0], thetaphi_off[i][:,1], None, 0)
        outfile = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/mock_maglim/photoz_unweighted/odmap_mock_maglim_photoz_nowgt_distMpc{:d}_{:d}_0arcmin.fits".format(binmin, binmax)
        print("Writing map to %s" %outfile)
        hp.write_map(outfile, odmap, overwrite=False)
 
