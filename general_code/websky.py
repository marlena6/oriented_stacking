import numpy as np
import healpy as hp
import h5py
from   cosmology import *

def read_halos(filepath, min_mass=None, max_mass=None):
    # takes npy file with format [M,x,y,z,ra,dec,redshift]                                                                     
    # returns ra, dec, comoving distance, mass                                                                                 
    halos = np.load(filepath)
    M   = halos[:,0]
    if min_mass != None and max_mass != None:
        include = (M > min_mass) & (M < max_mass)
        M   = M[include]
        x   = halos[:,1][include]
        y   = halos[:,2][include]
        z   = halos[:,3][include]
        ra  = halos[:,4][include]
        dec = halos[:,5][include]
        rshift   = halos[:,6][include]
    elif min_mass != None and max_mass == None:
        include = (M > min_mass)
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

def read_galcat(filepath, min_mass=10**13.5, max_mass=None, satfrac=1):
    import sys
    galcat = h5py.File(filepath, 'r')
    mcen = galcat['mass_cen'][()]
    print("read central masses, minimum central halo mass is ", np.amin(mcen))
    # print("Size of mcen list in memory", sys.getsizeof(mcen)) # is over 3 GB
    msat = galcat['mass_halo_sat'][()] # size of larger halo that satellite subhalo is within
    print("read satellite masses, minimum satellite halo mas is", np.amin(msat))
    # print("Size of msat list in memory", sys.getsizeof(msat)) # is nearly 15 GB
    if min_mass!= None or max_mass!= None: # if there is some mass constraint:
        if min_mass!=None and max_mass == None:
            satmass = (msat>min_mass)
            cenmass = (mcen>min_mass)
        elif min_mass==None and max_mass != None:
            satmass = (msat<max_mass)
            cenmass = (mcen<max_mass)
        elif min_mass!=None and max_mass!=None:
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
    else: # no mass constraint
        xsat = (galcat['xpos_sat'][()])
        ysat = (galcat['ypos_sat'][()])
        zsat = (galcat['zpos_sat'][()])
        xcen = (galcat['xpos_cen'][()])
        ycen = (galcat['ypos_cen'][()])
        zcen = (galcat['zpos_cen'][()])
    galcat.close()
    print("Size of msat list in memory", sys.getsizeof(msat))
    if satfrac!=1:
        Nsat = int(len(mcen)/((1/satfrac)-1))
        print("number of satellites", Nsat, "percentage", (Nsat/(Nsat+len(mcen))))
        satidx = np.arange(len(msat))
        newsats = np.random.choice(satidx, size=Nsat, replace=False)
        msat = msat[newsats]
        xsat = xsat[newsats]
        ysat = ysat[newsats]
        zsat = zsat[newsats]
        print("Size of msat list in memory", sys.getsizeof(msat))
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

def pksc_to_npy(filepath, colossus_cosmo, min_mass=10**10):
    astropy_cosmo = colossus_cosmo.toAstropy()
    from astropy.cosmology import z_at_value
    import astropy.units as u

    pkfile       = open(filepath)
    rho_m = colossus_cosmo.rho_m(0)*1000**3*h**2 #matter density in units of Msun h^2/kpc^3, converted to Msun h^2/Mpc^3
    Nhalo        = np.fromfile(pkfile, dtype=np.int32, count=3)[0]
    print("Number of halos:%d\n" %Nhalo)

    catalog = np.fromfile(pkfile,count=Nhalo*11,dtype=np.float32)
    catalog = np.reshape(catalog,(Nhalo,11))

    x  = catalog[:,0];  y = catalog[:,1];  z = catalog[:,2] # Mpc (comoving)
    vx = catalog[:,3]; vy = catalog[:,4]; vz = catalog[:,5] # km/sec
    R  = catalog[:,6] # Mpc

    # convert to mass, comoving distance, radial velocity, redshfit, RA and DEc
    M200m    = 4*np.pi/3.*rho_m*R**3      # this is M200m (mean density 200 times mean) in Msun*h^2, because R is in Lagrangian coordinates
    # this does neglect the delta_crit correction, but this is very small if doing lambdaCDM cosmologies between OmegaM = 0.2 to 0.4
    chi      = np.sqrt(x**2+y**2+z**2)    # Mpc
    vrad     = (x*vx + y*vy + z*vz) / chi # km/sec
    print("getting redshifts")
    redshift = z_at_value(astropy_cosmo.comoving_distance, chi[include]*u.Mpc) # `calculate after cut for speed
    print("finished with redhsifts")
    
    pkfile.close()

    print("done reading\n")

    ra,dec  = hp.vec2ang(np.column_stack((x,y,z)),lonlat=True)
    
    # reduce to only halos with mass greater than min_mass
    include = M200m>min_mass
    save_info = np.zeros((len(M200m[include]), 7))
    save_info[:,0] = M200m[include]
    save_info[:,1] = x[include]
    save_info[:,2] = y[include]
    save_info[:,3] = z[include]
    save_info[:,4] = ra[include]
    save_info[:,5] = dec[include]
    save_info[:,6] = redshift
    outfile = filepath.split('.')[0]+'.npy'
    np.save(outfile, save_info)

def pksc_to_hdf5(filepath, colossus_cosmo, min_mass=10**10):
    from colossus.halo import mass_defs, mass_adv
    import matplotlib.pyplot as plt
    import astropy.units as u
    astropy_cosmo = colossus_cosmo.toAstropy()
    from astropy.cosmology import z_at_value
    pkfile       = open(filepath)
    rho_m = colossus_cosmo.rho_m(0)*1000**3*h**2 #matter density in units of Msun h^2/kpc^3, converted to Msun h^2/Mpc^3
    Nhalo        = np.fromfile(pkfile, dtype=np.int32, count=3)[0]
    print("Number of halos:%d\n" %Nhalo)

    catalog = np.fromfile(pkfile,count=Nhalo*11,dtype=np.float32)
    catalog = np.reshape(catalog,(Nhalo,11))
    x  = catalog[:,0];  y = catalog[:,1];  z = catalog[:,2] # Mpc (comoving)
    vx = catalog[:,3]; vy = catalog[:,4]; vz = catalog[:,5] # km/sec
    R  = catalog[:,6] # Mpc

    # convert to mass, comoving distance, radial velocity, redshfit, RA and DEc
    M200m    = 4*np.pi/3.*rho_m*R**3      # this is M200m (mean density 200 times mean) in Msun*h^2, because R is in Lagrangian coordinates
    # this does neglect the delta_crit correction, but this is very small if doing lambdaCDM cosmologies between OmegaM = 0.2 to 0.4
    chi      = np.sqrt(x**2+y**2+z**2)    # Mpc
    vrad     = (x*vx + y*vy + z*vz) / chi # km/sec
    pkfile.close()

    print("done reading\n")

    ra,dec  = hp.vec2ang(np.column_stack((x,y,z)),lonlat=True)
    
    # reduce to only halos with mass greater than min_mass
    include = M200m>min_mass
    print("getting redshifts")
    redshift = z_at_value(astropy_cosmo.comoving_distance, chi[include]*u.Mpc) # calculate after cut for speed
    print("finished with redhsifts")
    # convert to M200c
    M200c = []
    for i in range(len(redshift)):
        M200c.extend(mass_adv.changeMassDefinitionCModel(M200m[include][i], (redshift.value)[i], '200m', '200c',c_model='bhattacharya13'))

    # make an hdf5 file for writing
    outfile = filepath.split('.')[0]+'.hdf5'
    f = h5py.File(outfile, 'a')
    f["ra"] = ra[include].astype(np.float32)
    f["dec"] = dec[include].astype(np.float32)
    f["m200c"] = np.asarray(M200c).astype(np.float32)
    f["m200m"] = M200m[include].astype(np.float32)
    f["z"] = redshift.astype(np.float32)
    # f["pos"] = pos[include].astype(np.float32)
    f.close()