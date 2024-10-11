# merge output files if reduce_object_catalog_region.py was run with multiple processors              

import numpy as np
import sys
import os
from astropy.table import Table, vstack


if len(sys.argv) != 4:
    sys.exit("Usage: <mode: pkp or data> <filepath to original non-reduced object file> <region size: large or small> \n")
mode     = sys.argv[1]
catpath,catname = os.path.split(sys.argv[2])
reg_size = sys.argv[3]


final_obj_count = 0
if mode == 'pkp':
    merged_filename = os.path.join(catpath, (("%s_region_" %reg_size) + "_abundance_matched_logMlt13_" + (os.path.splitext(catname)[0])+".npy"))
elif mode == 'data':
    merged_filename = os.path.join(catpath, (("%s_region_" %reg_size) + catname))
else:
    sys.exit("invalid mode.\n")
if os.path.exists(merged_filename):
    sys.exit("The file you are trying to create already exists.\n")
else:
    segfiles = [os.path.join(catpath,filename) for filename in os.listdir(catpath) if "seg" in filename]
    if segfiles[0].endswith("npy"):
        merged_dataArray = []
        with open(merged_filename, 'wb') as merged_file:
            for segment in segfiles:
                dataArray = np.load(segment)
                print(dataArray.shape)
                merged_dataArray.append(dataArray)
                final_obj_count += len(dataArray)
            merged_dataArray = np.concatenate(merged_dataArray, axis=0)
            print(merged_dataArray.shape)
            np.save(merged_file, merged_dataArray)
    elif segfiles[0].endswith("fit") or segfiles[0].endswith("fits"):

        # Read in the large table you want to append to 
        concat_table = Table.read(segfiles[0], format='fits')
        tables = []
        for segment in segfiles[1:]:
            concat_table = vstack([concat_table, Table.read(segment, format='fits')])
    
        concat_table.write(merged_filename, format='fits', overwrite=True)
        final_obj_count = len(concat_table)

        print("Remaining objects: %d\nSaved to %s\n" %(final_obj_count, merged_filename))
