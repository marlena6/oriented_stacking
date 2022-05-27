import numpy as np
import healpy as hp
import h5py
from   cosmology import *

def read_halos(filepath, min_mass=None, max_mass=None):
    # takes npy file with format [M,x,y,z,ra,dec,redshift]                                                                     
    # returns ra, dec, comoving distance, mass                                                                                 
    halos = np.load(filepath)
    M   = halos[:,0]
    if min_mass != None:
        include = (M > min_mass) & (M < max_mass)
        M   = M[include]
        x   = halos[:,1][include]
        y   = halos[:,2][include]
        z   = halos[:,3][include]
        ra  = halos[:,4][include]
        dec = halos[:,5][include]
        rshift   = halos[:,6][include]
    else:
        x   = halos[:,1]
        y   = halos[:,2]
        z   = halos[:,3]
        ra  = halos[:,4]
        dec = halos[:,5]
        rshift = halos[:,6]

    chi = np.sqrt(x**2+y**2+z**2)

    return(ra, dec, rshift, chi, M)

def galcat_length(filepath):
    return h5py.File(filepath, 'r')['mass_cen'].len()

def read_galcat(filepath, min_mass=10**13.5, max_mass=10**15, satfrac=1):
    galcat = h5py.File(filepath, 'r')
    mcen = galcat['mass_cen'][()]
    msat = galcat['mass_halo_sat'][()] # size of larger halo that satellite subhalo is within
    if min_mass!=None:
        satmass = (msat>min_mass) & (msat<max_mass)
        cenmass = (mcen>min_mass) & (mcen<max_mass)
        msat = msat[satmass]
        xsat = (galcat['xpos_sat'][()])[satmass]
        ysat = (galcat['ypos_sat'][()])[satmass]
        zsat = (galcat['zpos_sat'][()])[satmass]
        mcen = mcen[cenmass]
        xcen = (galcat['xpos_cen'])[cenmass]
        ycen = (galcat['ypos_cen'][()])[cenmass]
        zcen = (galcat['zpos_cen'][()])[cenmass]
    else:
        xsat = (galcat['xpos_sat'][()])
        ysat = (galcat['ypos_sat'][()])
        zsat = (galcat['zpos_sat'][()])
        xcen = (galcat['xpos_cen'][()])
        ycen = (galcat['ypos_cen'][()])
        zcen = (galcat['zpos_cen'][()])
    galcat.close()
    if satfrac!=1:
        Nsat = len(mcen)/((1/satfrac)-1)
        print("number of satellites", Nsat, "percentage", (Nsat/(Nsat+len(mcen))))
        satidx = np.arange(len(msat))
        newsats = np.random.choice(satidx, size=Nsat, replace=False)
        msat = msat[newsats]
        rasat = rasat[newsats]
        decsat = decsat[newsats]
        xsat = xsat[newsats]
        ysat = ysat[newsats]
        zsat = zsat[newsats]
    chicen = np.sqrt(xcen**2 + ycen**2 + zcen**2)
    chisat = np.sqrt(xsat**2 + ysat**2 + zsat**2)
    rshift_cen = zofchi(chicen)
    rshift_sat = zofchi(chisat)
    rasat,decsat   = hp.vec2ang(np.column_stack((xsat,ysat,zsat)), lonlat=True) # in degrees
    racen,deccen   = hp.vec2ang(np.column_stack((xcen,ycen,zcen)), lonlat=True) # in degrees
    if len(msat)>0:
        mass = np.concatenate((mcen, msat))
        ra   = np.concatenate((racen, rasat))
        dec  = np.concatenate((deccen, decsat))
        chi  = np.concatenate((chicen, chisat))
        rshift = np.concatenate((rshift_cen, rshift_sat))
    else:
        mass = mcen
        ra   = racen
        dec  = deccen
        chi  = chicen
        rshift = rshift_cen
    return(ra, dec, rshift, chi, mass)
