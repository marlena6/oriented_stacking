# combine the files from masking clusters into one
import sys
import os
from astropy.io import fits
from astropy.table import Table, vstack

catpath = sys.argv[1]
catname = sys.argv[2]
nfile   = int(sys.argv[3])
for n in range(nfile):
    if n==0:
        basefile = catpath + ("/small_region_seg%s_" %n)+catname
        base = fits.open(basefile)
        base_table = Table(base[1].data)
        print("Got base table shape from %s" %basefile)
        print("Length of base table: %d" %len(base_table))
    else:
        file_n   = catpath + ("/small_region_seg%s_" %n)+catname
        segment_n = fits.open(file_n)
        segment_table = Table(segment_n[1].data)
        base_table = vstack([base_table, segment_table])
        segment_n.close()
        print("Added segment %d to base table" %n)
        print("Length now: %d" %len(base_table))
        
base_table.write(catpath + "/small_region_" + catname, format='fits', overwrite=True)
base.close()
print("Wrote all segments to %s" %catpath + "/small_region_" + catname)