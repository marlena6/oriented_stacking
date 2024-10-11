import numpy as np
import coop_setup_funcs as csf
from astropy.cosmology import Planck18 as cosmo, z_at_value
import astropy.units as u
import subprocess

minz = 0.3
maxz = 0.6

#mode = 'unweighted_photoz'
sim = 'buzzard'
# mode = 'weighted_photoz'
# mode = 'ideal'
mode = 'lrgbin'
new_ini_file = "pkfile_mockredmapper_mockmaglim_{:s}_20Mpcsmooth.ini".format(mode)
if sim=='buzzard':
    testz_path = "/mnt/raid-cita/mlokken/buzzard/testing_photoz/"
elif sim=='websky':
    testz_path = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/mock_maglim/"
dlist     = csf.dlist(minz=minz, maxz=maxz, slice_width=200)
dlist_off = csf.dlist(minz=minz, maxz=maxz, slice_width=200, offset=100)
if mode in  ['weighted_photoz', 'unweighted_photoz', 'ideal']:
    for i in range(len(dlist)):
        dbin = dlist[i]
        binmin,binmax = dbin[0], dbin[1]
        bincent  = (binmin+binmax)/2.
        lower_z  = z_at_value(cosmo.comoving_distance, (bincent-50)*u.Mpc).value
        upper_z  = z_at_value(cosmo.comoving_distance, (bincent+50)*u.Mpc).value
        print("Finding clusters within 50 cMpc of {:.0f} Mpc".format(bincent))
        print("In redshift space, this is between {:.2f} and {:.2f}.".format(lower_z,upper_z))
        if sim=='websky':
            clfile = "/mnt/scratch-lustre/mlokken/pkpatch/testing_photozs/redmapper/thetaphi_100Mpc_centerofbin_distMpc_{:d}_{:d}.txt".format(binmin, binmax)
            
        elif sim=='buzzard':
            clfile = "/mnt/raid-cita/mlokken/buzzard/testing_photoz/redmapper/thetaphi_100Mpc_centerofbin_distMpc_{:d}_{:d}.txt".format(binmin,binmax)
            
        if mode=='weighted_photoz':
            odmapfile = testz_path + "photoz_weighted/odmap_mock_maglim_photoz_mult_distMpc{:d}_{:d}_0arcmin.fits".format(binmin, binmax)
        elif mode=='unweighted_photoz':
            odmapfile = testz_path + "photoz_unweighted/odmap_mock_maglim_photoz_nowgt_distMpc{:d}_{:d}_0arcmin.fits".format(binmin, binmax)
        elif mode=='ideal':
            if sim=='websky':
                odmapfile = testz_path+"true_z/odmap_1e+12_1e+16_distMpc{:d}_{:d}.fits".format(binmin, binmax)
            if sim=='buzzard':
                odmapfile = testz_path+"truez/odmap_mock_maglim_truez_distMpc{:d}_{:d}_0arcmin.fits".format(binmin,binmax)
        fout = open(new_ini_file, 'w')
        with open("pkfile_standard.ini", 'r') as file:
            for line in file:
                if line.startswith("map ="):
                    line = line.rstrip()
                    line += " "+odmapfile + '\n'
                    fout.write(line)
                elif line.startswith("output ="):
                    line = line.rstrip()
                    if sim=='websky':
                        line += "/mnt/scratch-lustre/mlokken/pkpatch/testing_photozs/"+"pksfile_{:s}_distMpc_{:d}_{:d}.fits".format(mode,binmin, binmax)
                    elif sim=='buzzard':
                        line += testz_path+"pks/pksfile_{:s}_distMpc_{:d}_{:d}.fits".format(mode,binmin, binmax)
                        fout.write(line+"\n")
                    print(line)
                elif line.startswith("external_list"):
                    line = line.rstrip()
                    line += " "+clfile
                    fout.write(line+"\n")
                    print(line)
                else:
                    fout.write(line)
        print("Running GetPeaks on {:s}".format(new_ini_file))
        subprocess.run(args=["/home/mlokken/software/COOP/mapio/GetPeaks",new_ini_file])

    for i in range(len(dlist_off)-1):
        dbin = dlist_off[i]
        binmin,binmax = dbin[0], dbin[1]
        bincent  = (binmin+binmax)/2.
        lower_z  = z_at_value(cosmo.comoving_distance, (bincent-50)*u.Mpc).value
        upper_z  = z_at_value(cosmo.comoving_distance, (bincent+50)*u.Mpc).value
        print("Finding clusters within 50 cMpc of {:.0f} Mpc".format(bincent))
        print("In redshift space, this is between {:.2f} and {:.2f}.".format(lower_z,upper_z))
        if sim=='websky':
            clfile = "/mnt/scratch-lustre/mlokken/pkpatch/testing_photozs/redmapper/thetaphi_100Mpc_centerofbin_distMpc_{:d}_{:d}.txt".format(binmin, binmax)
            testz_path = "/mnt/scratch-lustre/mlokken/pkpatch/number_density_maps/fullsky/galaxies/mock_maglim/"
        elif sim=='buzzard':
            clfile = "/mnt/raid-cita/mlokken/buzzard/testing_photoz/redmapper/thetaphi_100Mpc_centerofbin_distMpc_{:d}_{:d}.txt".format(binmin,binmax)
            testz_path = "/mnt/raid-cita/mlokken/buzzard/testing_photoz/"
        if mode=='weighted_photoz':
            odmapfile = testz_path + "photoz_weighted/odmap_mock_maglim_photoz_mult_distMpc{:d}_{:d}_0arcmin.fits".format(binmin, binmax)
        elif mode=='unweighted_photoz':
            odmapfile = testz_path + "photoz_unweighted/odmap_mock_maglim_photoz_nowgt_distMpc{:d}_{:d}_0arcmin.fits".format(binmin, binmax)
        elif mode=='ideal':
            if sim=='websky':
                odmapfile = testz_path+"true_z/odmap_1e+12_1e+16_distMpc{:d}_{:d}.fits".format(binmin, binmax)
            if sim=='buzzard':
                odmapfile = testz_path+"truez/odmap_mock_maglim_truez_distMpc{:d}_{:d}_0arcmin.fits".format(binmin,binmax)
        fout = open(new_ini_file, 'w')
        with open("pkfile_standard.ini", 'r') as file:
            for line in file:
                if line.startswith("map ="):
                    line = line.rstrip()
                    line += " "+odmapfile + '\n'
                    fout.write(line)
                elif line.startswith("output ="):
                    line = line.rstrip()
                    if sim=='websky':
                        line += "/mnt/scratch-lustre/mlokken/pkpatch/testing_photozs/"+"pksfile_{:s}_distMpc_{:d}_{:d}.fits".format(mode,binmin, binmax)
                    elif sim=='buzzard':
                        line += testz_path+"pks/pksfile_{:s}_distMpc_{:d}_{:d}.fits".format(mode,binmin, binmax)
                        fout.write(line+"\n")
                    print(line)
                elif line.startswith("external_list"):
                    line = line.rstrip()
                    line += " "+clfile
                    fout.write(line+"\n")
                    print(line)
                else:
                    fout.write(line)
        print("Running GetPeaks on {:s}".format(new_ini_file))
        subprocess.run(args=["/home/mlokken/software/COOP/mapio/GetPeaks",new_ini_file])

elif mode=='lrgbin':
    if sim=='websky':    
        clfile    = "/mnt/scratch-lustre/mlokken/pkpatch/testing_photozs/redmapper/thetaphi_deltaz_pt15_0.31_0.45.txt"
    elif sim=='buzzard':
        clfile    = testz_path + 'redmapper/thetaphi_deltaz_pt15_0.31_0.45.txt'
    fout = open(new_ini_file, 'w')
    with open("pkfile_standard.ini", 'r') as file:
        for line in file:
            if line.startswith("map ="):
                line = line.rstrip()
                line += odmapfile + '\n'
                fout.write(line)
            elif line.startswith("output ="):
                line = line.rstrip()
                if sim=='websky':
                    line += "/mnt/scratch-lustre/mlokken/pkpatch/testing_photozs/"+"pksfile_pzunwgt_0.31_0.45.fits"
                elif sim=='buzzard':
                    line += testz_path+"pks/pksfile_{:d}_truez_distMpc_{:d}_{:d}.fits"+"pksfile_pzunwgt_0.31_0.45.fits"
                    fout.write(line+"\n")
                print(line)
            elif line.startswith("external_list"):
                line = line.rstrip()
                line = line.replace(clfile)
                fout.write(line+"\n")
                print(line)
            else:
                fout.write(line)
    subprocess.run(args=["/home/mlokken/software/COOP/mapio/GetPeaks",new_ini_file])
