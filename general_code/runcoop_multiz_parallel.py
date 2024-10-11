import os
import sys
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u
from pathlib import Path
import subprocess
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if len(sys.argv)!=18:
    sys.exit("USAGE: <Full filename (including path) of the sample GetPeaks .ini file> <Full filename (including path) of the sample Stack .ini file> <Path to files of cluster (theta,phi)> <Path to number density map> <orientation mode: HESSIAN or QU> <smoothing scale [Mpc] (optional, type 'none' if no additional smoothing desired)> <nu min: none or float> <e-min: none or a float> <e-max: none or an integer> <symmetry type: SYMMETRIC, X_UP, Y_UP, XY_UP> <lambda minimum> <lambda maximum> <mode: pkp_y, data_y, pkp_ndmap, data_ndmap> <path to directory where Stacks folder and Peaks folder are (optional, type 'none' if this is in same location as .ini files)> <dmin> <dmax> <slice width[Mpc]>\n")

ini_file_pk  = sys.argv[1]
ini_file_stk = sys.argv[2]
cluster_path = sys.argv[3]
nd_path      = sys.argv[4]
orient_mode  = sys.argv[5]
smth_scale   = sys.argv[6]
nu_min       = sys.argv[7]
e_min        = sys.argv[8]
e_max        = sys.argv[9]
symmetry     = sys.argv[10]
lambda_min   = sys.argv[11]
lambda_max   = sys.argv[12]
data_mode    = sys.argv[13]
outpath      = sys.argv[14]
dmin         = int(sys.argv[15])
dmax         = int(sys.argv[16])
width        = int(sys.argv[17])

if data_mode not in ("pkp_y", "data_y", "pkp_ndmap", "data_ndmap"):
    sys.exit("Invalid mode entered.\n")

if outpath == 'none' or outpath == 'None':
    outpath = None
    path_to_peaks = os.path.join(os.path.dirname(ini_file_pk).rsplit('/',1)[0],'peaks/')
    path_to_stacks = os.path.join(os.path.dirname(ini_file_stk).rsplit('/',1)[0],'stacks/')

dbins = []
for i in range(dmin, dmax, width):
    dbins.append([i, i+200])

nruns_local = len(dbins) // size

if rank == size-1:
    extras = nruns_local % size
else:
    extras = 0

for i in range(nruns_local+extras):
    dbin = dbins[i+nruns_local*rank]
    lowerlim = dbin[0]
    upperlim = dbin[1]
    for ndmap in os.listdir(nd_path):
        if ndmap.endswith("arcmin.fits") and str(lowerlim) in ndmap and str(upperlim) in ndmap:
            name,ext = os.path.splitext(ndmap)
            namespl  = name.split('_')
            fwhm = namespl[-1].strip('arcmin')
            fwhm_in_Mpc = namespl[-2]
            mid = (float(lowerlim) + float(upperlim))/2. * u.megaparsec
            # get the redshift at the middle of this slice
            z = z_at_value(cosmo.comoving_distance, mid)
            # get the angular size (function gives arcseconds per kpc, convert to          
            # Mpc, then multiply by user-input scale [in Mpc]
            if smth_scale != 'none':
                smth_scale_arcmin = (cosmo.arcsec_per_kpc_comoving(z).to(u.arcsec/u.megaparsec)*smth_scale*u.Mpc).to(u.arcmin)
                if '.' in smth_scale:
                    smth_scale_str = smth_scale.replace('.', 'pt')
                else:
                    smth_scale_str = smth_scale
                ending = smth_scale_str + "Mpc_smth"
                #str(int(smth_scale_arcmin.value)) + 'a' + '_' 
            else:
                print("no smoothing scale entered. FWHM in Mpc found from number density file: %s \n"%fwhm_in_Mpc)
                ending = fwhm_in_Mpc + '_smth'

            fout_name = os.path.splitext(ini_file_pk)[0]+"_mod_%d_%d.ini" %(rank, i)
            if fout_name not in os.listdir(os.path.dirname(ini_file_pk)):
                Path(fout_name).touch()
            fout = open(fout_name, 'w')
            with open(ini_file_pk, 'r') as file:
                for line in file:
                    if line.startswith("map ="):
                        fout.write(line.replace(line, "map = %s" %os.path.join(nd_path,ndmap)))
                    elif line.startswith("fwhm ="):
                        fout.write(line.replace(line, "fwhm = %s." %fwhm))
                    elif line.startswith("output ="):
                        inifile_base = os.path.splitext(os.path.basename(ini_file_pk))[0]
                        if e_min != 'none':
                            inifile_base = inifile_base + "_e%sto%s" %(e_min.replace('.','pt'),e_max.replace('.','pt'))
                        if nu_min != 'none':
                            inifile_base = inifile_base + "_nugt%s" %(nu_min.replace('.','pt'))
                        inifile_base = inifile_base.replace("oriented", orient_mode+"_" + symmetry + "on%s" %ending)
                        if "redmagic" not in nd_path:
                            inifile_base = inifile_base.replace("redmagic", "cmass")
                        inifile_base = inifile_base.replace("general", "%sto%sMpc" %(lowerlim,upperlim))
                        inifile_base = inifile_base.replace("rm", "rmlambda%sto%s" %(lambda_min, lambda_max))
                        inifile_base = inifile_base + ".fits"
                        if not outpath:
                            fout.write(line.replace(line, "output = %s" %os.path.join(path_to_peaks,inifile_base)))
                        else:
                            path_to_peaks = os.path.join(outpath, "peaks")
                            if not os.path.exists(path_to_peaks):
                                try:
                                    os.mkdir(path_to_peaks)
                                except:
                                    print("Path to peaks already exists. Moving on.")
                            fout.write(line.replace(line, "output = %s" %os.path.join(path_to_peaks, inifile_base)))
                        print(path_to_peaks)
                        if os.path.exists(os.path.join(path_to_peaks,inifile_base)):
                            print("Peaks file already exists. Moving on to Stack.\n")
                            break
                    elif line.startswith("external_list ="):
                        if data_mode in ('pkp_y', 'pkp_ndmap'):
                            ext_list = [tpfile for tpfile in os.listdir(cluster_path) if (tpfile.endswith("%s_%s.txt" %(lowerlim,upperlim)) and tpfile.startswith('thetaphi'))]
                            fout.write(line.replace(line, "external_list = %s" %os.path.join(cluster_path,ext_list[0])))
                        elif data_mode in ('data_y', 'data_ndmap'):
                            ext_list = [tpfile for tpfile in os.listdir(cluster_path) if (tpfile.endswith("%s_%s.txt" %(lowerlim,upperlim)) and tpfile.startswith('thetaphi'))]
                            fout.write(line.replace(line, "external_list = %s" %os.path.join(cluster_path, ext_list[0])))
                        if os.stat(os.path.join(cluster_path, ext_list[0])).st_size == 0:
                            sys.exit("no stacking points in external list.")
                    elif line.startswith("orient ="):
                        fout.write(line.replace(line, "orient = %s" %orient_mode))
                    elif line.startswith("fwhm_orient"):
                        if smth_scale != 'none':
                            fout.write(line.replace(line, "fwhm_orient = %s\n" %str(int(smth_scale_arcmin.value))))
                        else:
                            fout.write(line.replace(line, "fwhm_orient = 0.\n"))
                    elif line.startswith("e_min ="):
                        if e_min != 'none':
                            fout.write(line.replace(line, "e_min = %s\n" %e_min))
                    elif line.startswith("e_max ="):
                        if e_max != 'none':
                            fout.write(line.replace(line, "e_max = %s\n" %e_max))
                    elif line.startswith("fwhm_e ="):
                        if smth_scale != 'none':
                            fout.write(line.replace(line, "fwhm_e = %s\n" %str(int(smth_scale_arcmin.value))))
                        else:
                            fout.write(line.replace(line, "fwhm_e = 0.\n"))
                    elif line.startswith("nu_min ="):
                        if nu_min != 'none':
                            fout.write(line.replace(line, "nu_min = %s\n" %nu_min))
                    elif line.startswith("fwhm_nu ="):
                        if smth_scale != 'none':
                            fout.write(line.replace(line, "fwhm_nu = %s\n" %str(int(smth_scale_arcmin.value))))
                        else:
                            fout.write(line.replace(line, "fwhm_nu = 0.\n"))
                    elif line.startswith("symmetry ="):
                        fout.write(line.replace(line, "symmetry = %s\n" %symmetry))
                    elif line.startswith("sym_option = "):
                        fout.write(line.replace(line, "sym_option = %s\n" %orient_mode))
                    elif line.startswith("fwhm_sym ="):
                        if smth_scale != 'none':
                            fout.write(line.replace(line, "fwhm_sym = %s\n" %str(int(smth_scale_arcmin.value))))
                    else:
                        fout.write(line)
            fout.close()

            # modify stacking file
            fout_stk_name = os.path.splitext(ini_file_stk)[0]+"_mod_%d_%d.ini" %(rank, i)
            if fout_stk_name not in os.listdir(os.path.dirname(ini_file_stk)):
                Path(fout_stk_name).touch()
            fout_stk = open(fout_stk_name, 'w')
            with open(ini_file_stk, 'r') as file:
                for line in file:
                    # Check if the mode is one where we stack on the number density map.
                    # If so, update the map line to be that map. Otherwise, map should already be set to the correct one and won't be changed.
                    if line.startswith("map = ") and (data_mode in ("pkp_ndmap", "data_ndmap")):
                        if smth_scale == 'none':
                            fout_stk.write(line.replace(line, "map = %s\n" %os.path.join(nd_path,ndmap)))
                        else:
                            sys.exit("Pre-smooth number density maps or figure out how to use the COOP-generated ones.\n")
                    if line.startswith("peaks = "):
                        fout_stk.write(line.replace(line, "peaks = %s\n" %os.path.join(path_to_peaks,inifile_base)))
                    elif line.startswith("output = "):
                        inifile_stk_base = os.path.splitext(os.path.basename(ini_file_stk))[0]
                        if e_min != 'none':
                            inifile_stk_base = inifile_stk_base + "_e%sto%s" %(e_min.replace('.','pt'),e_max.replace('.','pt'))
                        if nu_min != 'none':
                            inifile_stk_base = inifile_stk_base + "_nugt%s" %(nu_min.replace('.','pt'))
                        inifile_stk_base = inifile_stk_base.replace("oriented", orient_mode+"_" + symmetry + "on%s" %ending)
                        inifile_stk_base = inifile_stk_base.replace("general", "%sto%sMpc" %(lowerlim,upperlim))
                        inifile_stk_base = inifile_stk_base.replace("rm", "rmlambda%sto%s" %(lambda_min, lambda_max))
                        if "redmagic" in nd_path:
                            inifile_stk_base = inifile_stk_base.replace("galaxyfield", "redmagic")
                        elif "buzz" in nd_path:
                            inifile_stk_base = inifile_stk_base.replace("galaxyfield", "buzzgals")
                        else:
                            inifile_stk_base = inifile_stk_base.replace("galaxyfield", "cmass")
                        if not outpath:
                            fout_stk.write(line.replace(line, "output = %s\n" %os.path.join(path_to_stacks, inifile_stk_base)))
                        else:
                            path_to_stacks = os.path.join(outpath, "stacks")
                            if not os.path.exists(path_to_stacks):
                                os.mkdir(path_to_stacks)
                            fout_stk.write(line.replace(line, "output = %s\n" %os.path.join(path_to_stacks, inifile_stk_base)))
                    else:
                        fout_stk.write(line)
            fout_stk.close()

         
            if not os.path.exists(os.path.join(path_to_peaks,inifile_base)):
                print('Peaks and Stacks .ini files are now updated. Running GetPeaks.\n')
                subprocess.run(args=["/home/mlokken/software/COOP/mapio/GetPeaks",fout_name])
            else:
                print('Stacks file updated.\n')
            print("Running Stack.\n")
            subprocess.run(args=["/home/mlokken/software/COOP/mapio/Stack",fout_stk_name])
            # remove new ini files after this is done.
            os.remove(fout_stk_name)
            os.remove(fout_name)
