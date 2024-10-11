import matplotlib.pyplot as plt
from pixell import enmap as em
from pixell import reproject as rp
from pixell import curvedsky
import numpy as np
from astropy.io import fits
import sys
import os
import healpy as hp

def actmap_to_healpix(infile, beam=None, smoothing=False, nside_downgrade=False, maxell_zeroed=None, is_mask=False):
    print(infile)
    inmap = em.read_fits(infile) # automatically hdu=0. Can also slice with sel
    if nside_downgrade:
        nside = nside_downgrade
    else:
        nside = 8192
    if smoothing == True:
        smoothmap = em.smooth_gauss(inmap,beam)
    else:
        smoothmap = inmap
    #Next 2 lines for Planck map only
    # temp_data = inmap[0,:,:] 
    # outmap = em.to_healpix(smoothmap, nside=nside)
    # outmap = em.to_healpix(smoothmap, destroy_input=True, nside=8192) #nside = 0 automatically: output map is higher res than input
    if nside_downgrade:
        lmax = 3*nside
    else:
        lmax = 30000
    if maxell_zeroed is not None:
        alm = curvedsky.map2alm(inmap, lmax=lmax, spin=0)
        if alm.ndim > 1:
            assert alm.shape[0] == 1
            alm = alm[0]
        fl  = np.ones(lmax)
        fl[0:maxell_zeroed]=0
        alm = hp.almxfl(alm, fl) # set modes up until that limit to 0
        outmap = hp.alm2map(alm.astype(np.complex128), nside, lmax=lmax)
    else:
        print('reprojecting.')
        if is_mask:
            outmap = rp.healpix_from_enmap_interp(smoothmap, nside=nside) # supposed to use this for masks to avoid ringing
        else:
            outmap = rp.healpix_from_enmap(smoothmap, lmax, nside)
    print("Transformed map to Healpix format with NSIDE=%f"%nside)
    return(outmap)
if len(sys.argv)<6 or len(sys.argv)>7:
    sys.exit("USAGE: <filename> <smooth map? (True or False)> <beam FWHM [arcmin] (None if not smoothing)> <nside for downgraded map (none if none needed)> <set alms to zero up to this ell (none otherwise)>\n")

mapname = sys.argv[1]
smoothing = sys.argv[2]
if smoothing == 'False':
    smoothing = False
if smoothing == 'True':
    smoothing = True
nside_final = sys.argv[4]
if nside_final not in ('none','None'):
    nside_final = int(nside_final)
else:
    nside_final = None
if smoothing == True:
    beam_arcmin = float(sys.argv[3])
    beam_rad = beam_arcmin / 60. * (np.pi/180.) # convert to radians
    beam_sigma = beam_rad /np.sqrt(2*np.log(2)) # convert from FWHM to sigma of the Gaussian beam
    print(beam_rad)
    print(beam_sigma)
else:
    beam_sigma = None
alm_lim = sys.argv[5]
if alm_lim in ('none','None'):
    alm_lim = None
else:
    alm_lim = int(alm_lim)
if len(sys.argv)==7:
    is_mask = True
else:
    is_mask = False
print(mapname)
print('starting function')
outmap = actmap_to_healpix(mapname, beam_sigma, smoothing, nside_final, alm_lim, is_mask)
if nside_final is not None:
    ending = '_{:d}_hpx.fits'.format(nside_final)
else:
    ending = '_hpx.fits'
hp.write_map(mapname[:-5]+ending, outmap, overwrite=True)
