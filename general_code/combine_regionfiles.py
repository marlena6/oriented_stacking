import numpy as np
import pickle
import os
import shutil
import coop_setup_funcs as csf
from astropy import units as u
# mode = 'GRF'                                                                                                                                                                                            
# mode  = 'Buzzard'
mode = 'ACTxDES'
# mode   = 'Websky'                                                                                                                                                                                       
# mode = "Cardinal"

if mode == 'ACTxDES':
    outpath     = "/mnt/scratch-lustre/mlokken/stacking/ACTxDES_paper2/"
    orient_mode = "maglim"
if mode == 'Buzzard':
    outpath = "/mnt/scratch-lustre/mlokken/stacking/Buzzard_paper2/"
    orient_mode = "maglim"
if mode == "Websky":
    outpath     = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/orient_tests/"
if mode == 'Cardinal':
    outpath = "/mnt/scratch-lustre/mlokken/stacking/Cardinal_paper2/"
    orient_mode = "maglim"

path = os.path.join(outpath, "temp_reg")
width    = 200
minz     = 0.2 # if you want to only run from the lower d limit of paper 1, input z_at_value(cosmo.comoving_distance, 1032.5*u.Mpc)
maxz     = 1.0
dlist     = csf.dlist(minz=minz, maxz=maxz, slice_width=200)
dlist_off = csf.dlist(minz=minz, maxz=maxz, slice_width=200, offset=100)
dlist_tot = [None]*(len(dlist)+len(dlist_off))
dlist_tot[::2] = dlist
dlist_tot[1::2] = dlist_off
cl_dlist = [[dbin[0]+50,dbin[1]-50] for dbin in dlist_tot]
# nlow_hi_bins = [[0,22]]  # for full list
nlow_hi_bins = [[0,4], [6,10], [12,16], [18,22]]
# nlow_hi_bins = [[2,16]]

for file in os.listdir(path):
    f = np.load(os.path.join(path,file), allow_pickle=True)
    keylist = f.keys()
    break


# combine the y stacks
for nbin in nlow_hi_bins:
    combining = False
    mydict = {}
    for key in keylist:
        mydict[key]=[]

    nlow = nbin[0]
    nhi  = nbin[1]
    cl_dlow_abs = cl_dlist[nlow][0]
    cl_dhi_abs  = cl_dlist[nhi][1]
    dlow_abs    = dlist_tot[nlow][0]
    dhi_abs     = dlist_tot[nhi][1]
    for file in os.listdir(path):
        # check if it's a Compton-y stack
        if (f"{cl_dlow_abs}_{cl_dhi_abs}") in file and (("yy" in file) or ("ymap" in file)):
            combining = True
            f = np.load(os.path.join(path,file), allow_pickle=True)
            for key in f.keys():
                if type(mydict[key])==np.ndarray: # the first append converted it to a numpy array
                    mydict[key] = mydict[key].tolist()    
                if type(f[key])==list and len(f[key])==1:
                    if type(f[key][0])==int:
                        mydict[key].extend(f[key])
                    else:
                        mydict[key].append(f[key][0])
                    
                elif type(f[key])==int or type(f[key])==u.quantity.Quantity:
                    mydict[key].append(f[key])
                else:
                    mydict[key].extend(f[key])
                # if type(mydict[key])==list:
                #     mydict[key] = mydict[key][0]
            first = '_'.join(file.split('_')[:-4])
            last  = '_'.join(file.split('_')[-2:])
            savestr = first+'_'+last
    if combining:
        save_file = open(os.path.join(outpath, savestr), "wb")
        pickle.dump(mydict, save_file)
        save_file.close()
        print(save_file)

# combine the g stacks
for nbin in nlow_hi_bins:
    combining = False
    mydict = {}
    for key in keylist:
        mydict[key]=[]

    nlow = nbin[0]
    nhi  = nbin[1]
    cl_dlow_abs = cl_dlist[nlow][0]
    cl_dhi_abs  = cl_dlist[nhi][1]
    dlow_abs    = dlist_tot[nlow][0]
    dhi_abs     = dlist_tot[nhi][1]

    print("dlow, dhi", dlow_abs, dhi_abs)
    for file in os.listdir(path):
        # check if it's a galaxy ndmap stack
        if (f"{cl_dlow_abs}_{cl_dhi_abs}") in file and (("DES_maglim" in file) or ("Buzzard_maglim" in file) or ("Cardinal_maglim" in file)):
            combining = True
            f = np.load(os.path.join(path,file), allow_pickle=True)
            for key in f.keys():
                if type(mydict[key])==np.ndarray: # the first append converted it to a numpy array
                    mydict[key] = mydict[key].tolist()
                if type(f[key])==list and len(f[key])==1:
                    if type(f[key][0])==int:
                        mydict[key].extend(f[key])
                    else:
                        mydict[key].append(f[key][0])
                elif type(f[key])==int or type(f[key])==u.quantity.Quantity:
                    mydict[key].append(f[key])
                else:
                    mydict[key].extend(f[key])
            first = '_'.join(file.split('_')[:-4])
            last  = '_'.join(file.split('_')[-2:])
            savestr = first+'_'+last
    if combining:
        save_file = open(os.path.join(outpath, savestr), "wb")
        pickle.dump(mydict, save_file)
        save_file.close()
        print(save_file)

# combine the kappa stacks
for nbin in nlow_hi_bins:
    combining = False
    mydict = {}
    for key in keylist:
        mydict[key]=[]

    nlow = nbin[0]
    nhi  = nbin[1]
    cl_dlow_abs = cl_dlist[nlow][0]
    cl_dhi_abs  = cl_dlist[nhi][1]
    dlow_abs    = dlist_tot[nlow][0]
    dhi_abs     = dlist_tot[nhi][1]
    for file in os.listdir(path):
        # check if it's a kappa stack
        if (f"{cl_dlow_abs}_{cl_dhi_abs}") in file and (("kappa_bin4_asdelta" in file)):
            combining = True
            f = np.load(os.path.join(path,file), allow_pickle=True)
            for key in f.keys():
                if type(mydict[key])==np.ndarray: # the first append converted it to a numpy array
                    mydict[key] = mydict[key].tolist()
                if type(f[key])==list and len(f[key])==1:
                    if type(f[key][0])==int:
                        mydict[key].extend(f[key])
                    else:
                        mydict[key].append(f[key][0])
                elif type(f[key])==int or type(f[key])==u.quantity.Quantity:
                    mydict[key].append(f[key])
                else:
                    mydict[key].extend(f[key])
                first = '_'.join(file.split('_')[:-4])
            last  = '_'.join(file.split('_')[-2:])
            savestr = first+'_'+last
    if combining:
        save_file = open(os.path.join(outpath, savestr), "wb")
        pickle.dump(mydict, save_file)
        save_file.close()
        print(save_file)

# combine the mask stacks
for nbin in nlow_hi_bins:
    combining = False
    mydict = {}
    for key in keylist:
        mydict[key]=[]

    nlow = nbin[0]
    nhi  = nbin[1]
    cl_dlow_abs = cl_dlist[nlow][0]
    cl_dhi_abs  = cl_dlist[nhi][1]
    dlow_abs    = dlist_tot[nlow][0]
    dhi_abs     = dlist_tot[nhi][1]
    for file in os.listdir(path):
        # check if it's a mask stack
        if (f"{cl_dlow_abs}_{cl_dhi_abs}") in file and (("mask" in file)):
            combining = True
            f = np.load(os.path.join(path,file), allow_pickle=True)
            for key in f.keys():
                if type(mydict[key])==np.ndarray: # the first append converted it to a numpy array
                    mydict[key] = mydict[key].tolist()
                if type(f[key])==list and len(f[key])==1:
                    if type(f[key][0])==int:
                        mydict[key].extend(f[key])
                    else:
                        mydict[key].append(f[key][0])
                elif type(f[key])==int or type(f[key]=='Quantity'):
                    mydict[key].append(f[key])
                else:
                    mydict[key].extend(f[key])
                first = '_'.join(file.split('_')[:-4])
            last  = '_'.join(file.split('_')[-2:])
            savestr = first+'_'+last
    if combining:
        save_file = open(os.path.join(outpath, savestr), "wb")
        pickle.dump(mydict, save_file)
        save_file.close()
        print(save_file)

# remove the temporary directory
#shutil.rmtree(outpath+"temp_reg")

