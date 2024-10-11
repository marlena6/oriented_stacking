import os
import sys
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u
from pathlib import Path
import subprocess
from mpi4py import MPI
import numpy as np

def get_R(z):
    # this is in physical units, I think
    rho_at_z = cosmo.critical_density(z)*cosmo.Odm(z)
    M = 10**15*u.Msun
    R = ((M/((2*np.pi)**(3/2.)*rho_at_z))**(1/3.)).to(u.Mpc)
    scale_arcmin = (cosmo.arcsec_per_kpc_proper(z).to(u.arcmin/u.Mpc))*R
    return(R, scale_arcmin)

h = (cosmo.H(0)/100.).value
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if size!=3:
    sys.exit("Wrong number of processors. Must be 3.\n")

if len(sys.argv)!=9:
    sys.exit("USAGE: <Full filename (including path) of the sample GetPeaks .ini file> <Full filename (including path) of the sample Stack .ini file> <peak finding scale (fwhm, physical Mpc)> <smoothing scale for nu, e (fwhm, comoving Mpc over h)> <Do thresholding? y or n> <mode> <threshpks y or n> <extpks y or n>\n")

ini_file_pk   = sys.argv[1]
ini_file_stk  = sys.argv[2]
pk_scale      = sys.argv[3]
smth_scale    = sys.argv[4]
thresholding  = sys.argv[5]
mode          = sys.argv[6]
threshpks     = sys.argv[7]
extpks        = sys.argv[8]

if thresholding in ['y', 'Y']:
    thresholding = True
else:
    thresholding = False

if threshpks in ['y', 'Y']:
    threshpks = True
else:
    threshpks = False

if extpks in ['y','Y']:
    extpks = True
else:
    extpks = False

def make_path(lst):
    path = ""
    for l in lst:
        path = path + l + "/"
    return(path)

outpath = make_path(ini_file_pk.split('/')[0:-2])
print(outpath)

zbins = [[0.15,0.35], [0.35,0.5], [0.5,0.65]]

zbin = zbins[rank]

print("z bin: ", zbin)
zmid = (zbin[0] + zbin[1])/2.
dhi  = cosmo.comoving_distance(zbin[1])
dlow = cosmo.comoving_distance(zbin[0])
dmid = (dhi+dlow)/2.

pk_scale_arcmin = cosmo.arcsec_per_kpc_proper(zmid).to(u.arcmin/u.megaparsec)*float(pk_scale)*u.Mpc
smth_scale_arcmin = cosmo.arcsec_per_kpc_comoving(zmid).to(u.arcmin/u.megaparsec)*float(smth_scale)/h*u.Mpc
print("Peak scale: ", pk_scale_arcmin)
print("Smth scale: ", smth_scale_arcmin)

# if thresholding == True:
#     R_threshold_Mpc, R_threshold_arcmin = get_R(zmid)
#     print("thresholding: ", R_threshold_Mpc, R_threshold_arcmin)
R_threshold_arcmin = smth_scale_arcmin

fout_name = os.path.splitext(ini_file_pk)[0]+"_mod_%d.ini" %(rank)

if fout_name not in os.listdir(os.path.dirname(ini_file_pk)):
    Path(fout_name).touch()
    fout = open(fout_name, 'w')
    with open(ini_file_pk, 'r') as file:
        for line in file:
            if line.startswith("map ="):
                if mode == 'grf':
                    if "mybuzzardyy" in ini_file_pk:
                        print("Buzzard GRF\n")
                        fout.write(line.replace(line, "map = " + os.path.join(outpath,"grf_galfield_pandeyfit_rmagic_mybuzzardyy_zbin{:d}.fits".format(rank+1))))
                    else:    
                        fout.write(line.replace(line, "map = " + os.path.join(outpath,"grf_galfield_pandeyfit_rmagic_act_zbin{:d}_wmask.fits".format(rank+1))))
                elif mode == 'buzzard':
                    fout.write(line.replace(line, "map = /mnt/scratch-lustre/mlokken/buzzard/number_density_maps/des_reg/3_zbins_grf_comparison/buzzard_redmagic_highdens_zbin{:d}_od_map_nosmooth.fits".format(rank+1)))
                elif mode == 'redmagic':
                    fout.write(line.replace(line, "map = /mnt/scratch-lustre/mlokken/data/number_density_maps/3_zbins_grf_comparison/redmagic_highdens_zbin{:d}_od_map_nosmooth.fits".format(rank+1)))
            elif line.startswith("output ="):
                inifile_base = os.path.splitext(os.path.basename(ini_file_pk))[0]
                inifile_base = inifile_base.replace("general", "zbin{:d}_".format(rank+1))
                if mode == 'grf':
                    inifile_base = inifile_base + 'pks_on_{:s}Mpc_'.format(pk_scale)
                if thresholding == True:
                    inifile_base = inifile_base + "nugt2_"
                inifile_base = inifile_base + 'smth_{:s}overhMpc'.format(smth_scale)
                inifile_base = inifile_base.replace(".", "pt")
                inifile_base = inifile_base + ".fits"
                fout.write(line.replace(line, "output = %s" %os.path.join(outpath, 'peaks',inifile_base)))
            elif line.startswith("external_list =") and mode == 'redmagic':
                fout.write(line.replace(line, "external_list = /mnt/scratch-lustre/mlokken/data/clusters_to_stack/y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_lgt5_vl50/3_zbins_grf_comparison/thetaphi_lambda_7_to_200_zbin{:d}.txt\n".format(rank+1)))
            elif line.startswith("external_list =") and mode == "buzzard":
                fout.write(line.replace(line, "external_list = /mnt/scratch-lustre/mlokken/buzzard/clusters_to_stack/buzzard_1.9.9_3y3a_rsshift_run_redmapper_v0.5.1_lgt05_vl50/des_reg/3_zbins_grf_comparison/thetaphi_lambda_7_to_200_zbin{:d}.txt\n".format(rank+1)))
            elif line.startswith("external_list =") and mode == "grf" and extpks == True:
                fout.write(line.replace(line, "external_list = /mnt/scratch-lustre/mlokken/stacking/GRF/zbin{:d}_externalpks_nugtpt5on1pt67Mpc.txt".format(rank+1)))
            elif line.startswith("nu_min ="):
                if thresholding == True:
                    fout.write(line.replace(line, "nu_min = 2\n"))
                elif threshpks == True:
                    fout.write(line.replace(line, "nu_min = 0.5\n"))
                else:
                    fout.write(line)
            elif line.startswith("e_min ="):
                if thresholding == True:
                    fout.write(line.replace(line, "e_min = 0.2\n"))
                else:
                    fout.write(line)
            elif line.startswith("fwhm_pt =") and mode == 'grf' and extpks == False:
                fout.write(line.replace(line, "fwhm_pt = {:f}\n".format(pk_scale_arcmin.value)))
            elif line.startswith("fwhm_orient"):
                fout.write(line.replace(line, "fwhm_orient = {:f}\n".format(smth_scale_arcmin.value)))
            elif line.startswith("fwhm_e"):
                fout.write(line.replace(line, "fwhm_e = {:f}\n".format(smth_scale_arcmin.value)))
            elif line.startswith("fwhm_nu") and thresholding == True:
                fout.write(line.replace(line, "fwhm_nu = {:f}\n".format(R_threshold_arcmin.value)))
            elif line.startswith("fwhm_nu") and threshpks == True:
                fout.write(line.replace(line, "fwhm_nu = {:f}\n".format(pk_scale_arcmin.value)))
            else:
                fout.write(line)
    fout.close()


fout_stk_name = os.path.splitext(ini_file_stk)[0]+"_mod_%d.ini" %(rank)   

if fout_stk_name not in os.listdir(os.path.dirname(ini_file_stk)):
    Path(fout_stk_name).touch()
    fout_stk = open(fout_stk_name, 'w')
    with open(ini_file_stk, 'r') as file:
        for line in file:
            if line.startswith("map =") and mode == 'grf' and 'mybuzzardyy' not in ini_file_stk:
                fout_stk.write(line.replace(line, "map = " + os.path.join(outpath,"grf_yfield_pandeyfit_rmagic_act_zbin{:d}_wmask.fits".format(rank+1))))
            elif line.startswith("map =") and mode == 'grf' and 'mybuzzardyy' in ini_file_stk:
                fout_stk.write(line.replace(line, "map = " + os.path.join(outpath,"grf_yfield_pandeyfit_rmagic_mybuzzardyy_zbin{:d}.fits".format(rank+1))))
            elif line.startswith("output ="):
                fout_stk.write(line.replace(line, "output = {:s}\n".format(os.path.join(outpath, 'stacks',(inifile_base[:-5]+"_stack\n").replace("gal", "y")))))
            elif line.startswith("peaks ="):
                fout_stk.write(line.replace(line, "peaks = {:s}\n".format(os.path.join(outpath, 'peaks',inifile_base))))
            else:
                fout_stk.write(line)
    fout_stk.close()

print(fout_name)
print("Running GetPeaks.\n")
subprocess.run(args=["/home/mlokken/software/COOP/mapio/GetPeaks",fout_name])
print("Running Stack.\n")
subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack", fout_stk_name])
