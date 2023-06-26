import os
import sys
import numpy as np
from astropy.io import fits
from astropy.table import Table
import healpy as hp
from   cosmology import *
from pixell import enmap,enplot,utils
import unittest
from mpi4py import MPI
import astropy.units as u
from pixell import enmap as em

nside = 4096

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#user message 
if len(sys.argv)!=3:
    sys.exit("USAGE: <path to object catalog (either clusters or galaxies)> <path to mask>")

catpath,catname = os.path.split(sys.argv[1])
mask_file = sys.argv[2]
catalog = fits.open(sys.argv[1])
hdr = catalog[1].header
data = catalog[1].data
catalog.close()
mask = em.read_map(mask_file)


print("Reading object catalog\n")
print("Total number of objects in this catalog: %d\n\n" %len(data))
    #table which will hold reduced catalog                                                                                                                                                                
t = Table(data)
if 'ra' in t.colnames:
    raname = 'ra'
    decname = 'dec'
else:
    raname = 'RA'
    decname = 'DEC'
    
ra = data[raname]
dec = data[decname]
print(ra[0], dec[0])

include_idx = np.arange(len(dec))
N = len(ra)

ncl_local = N // size
if rank == size - 1:
    extras = N % size
else:
    extras = 0

#find clusters within 2 degree of edge
if size != 1:                                                                                                                                                                                        
    print("Processor %d: looping through %d objects\n" %(rank, ncl_local+extras))
coopwidth = 2 #degrees
height,width = enmap.pixshape(mask.shape,mask.wcs) #radians
pixel_width = np.deg2rad(coopwidth)//width
print("Cutting clusters more than %d pixels from map edge"%pixel_width)

good_rows = []

for i in range(ncl_local+extras):
    clnum = i + ncl_local * rank
    coords = np.deg2rad(np.array((dec[clnum],ra[clnum])))
    ypix, xpix = enmap.sky2pix(mask.shape, mask.wcs, coords)
    
    left   = int(xpix - pixel_width)
    right  = int(xpix + pixel_width + 1)
    bottom = int(ypix - pixel_width)
    top    = int(ypix + pixel_width + 1)
    # Make a box of the mask values in the cutout around the cluster
    maski = mask[bottom:top, left:right]
    # check if the mean is low enough to indicate this is near the edge
    if maski.size == 0:
        mean = 0.
    else:
        mean = np.mean(maski)
    if mean<0.8:
        continue
    else:
        good_rows.append(include_idx[clnum])


if size == 1:
    savefile = catpath + ("/small_region_"+catname)
else:
    savefile = catpath + ("/small_region_seg%s_" %(rank))+catname
new_table = t[good_rows]
final_ra = new_table[raname]
final_dec = new_table[decname]
new_table.write(savefile,fits)
print("Saved table to %s\n"%savefile)
