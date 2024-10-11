import sys
import numpy as np
import websky as ws

mask_path = None # fullsky

# mask_path = "/mnt/raid-cita/mlokken/data/masks/y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask_hpx_4096.fits"
obj = 'clusters'                          
#obj = 'galaxies'
if obj=='clusters':          
    catfile = "/mnt/scratch-lustre/mlokken/pkpatch/halos_fullsky_M_gt_1E13.npy"
    outpath = "/mnt/scratch-lustre/mlokken/pkpatch/testing_photozs/redmapper/"
    masswgt_odmap = True # always set to True for halos                                                                                                                                               
if obj=='galaxies':
    catfile = "/mnt/scratch-lustre/mlokken/pkpatch/galaxy_catalogue.h5"
    outpath = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/mock_maglim/"

if obj == 'galaxies':
    min_mass = None # minimum Websky halo mass is 10^12 Msun anyway, and Maglim's is even lower than that. 
    max_mass = None
if obj == 'clusters':
    min_mass = 10**14 # around lambda=20 for redMaPPer
    max_mass = None

print("Reading catalog.")
if obj=='clusters':
    ra, dec, z, chi, mass = ws.read_halos(catfile, min_mass, max_mass)
elif obj=='galaxies':
    ra, dec, z, chi, mass = ws.read_galcat(catfile, min_mass, max_mass, satfrac=.15)
print("Catalog read.")

# reduce the redshift range
inz = (z>0.15)& (z<0.65) # most should fall in this range even after smearing
ra, dec, z, chi, mass = ra[inz], dec[inz], z[inz], chi[inz], mass[inz]
print("Size of reduced list in memory (div. 5)", sys.getsizeof(ra))
print("Time to photo-z smear")
# do the smearing
if obj=='galaxies':
    photoz = np.random.normal(z, 0.05*(1+z)) # maglim at z=.4
elif obj=='clusters':
    photoz = np.random.normal(z, 0.015*(1+z)) # redmapper
to_save = np.column_stack((ra, dec, photoz, z, chi, mass))
print("ra", ra[:10])
print("pz", photoz[:10])
print("array to save", to_save[:10])
print("Shape of array to save, should be (ngal, 6)", to_save.shape)
print("Size of to-be-saved array in memory (div. 5)", sys.getsizeof(to_save))

if obj=='galaxies':
    np.save(outpath+"mock_maglim_photoz.npy", np.column_stack((ra, dec, photoz, chi, mass)))
elif obj =='clusters':
    np.save(outpath + "mock_redmapper_photoz.npy", np.column_stack((ra, dec, photoz, chi, mass)))   
