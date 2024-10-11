import subprocess
import coop_post_processing as cpp
import shutil
import numpy as np
import os
'''
# Run GetPeaks for 20 Mpc smoothing, nu=[2,3], e=[0.3,0.4]
inifile = "theta_of_r_pkfile_20Mpc.ini"
subprocess.run(args=["/home/mlokken/software/COOP/mapio/GetPeaks",inifile])
'''
# Read output to get fixed sample of stacking points
#rot_angle_20, theta, phi = cpp.peakinfo_thetaphi("/mnt/scratch-lustre/mlokken/stacking/data_ndmap/theta_of_r/20Mpc_nu2to3_ept3topt4.fits")
rot_angle_20, theta, phi = cpp.peakinfo_thetaphi("/mnt/scratch-lustre/mlokken/stacking/PeakPatch_halomap/orient_by_1pt5E12_to_1E15_msun_halos/theta_of_r/20Mpc_nu2to3_ept3topt4.fits")
thetaphi = np.stack((theta,phi), axis=-1)
#np.savetxt("/mnt/scratch-lustre/mlokken/stacking/data_ndmap/theta_of_r/rm_cls_lgt20_z_0pt55_0pt7_nu2to3_ept3topt4_20Mpc.txt",thetaphi)
np.savetxt("/mnt/scratch-lustre/mlokken/stacking/PeakPatch_halomap/orient_by_1pt5E12_to_1E15_msun_halos/theta_of_r/rm_cls_lgt20_z_1600_1800Mpc_nu2to3_ept3topt4_20Mpc.txt",thetaphi)

# for scale in [5,10,15,25,30,35, 40]: # ,40,45
for scale in [40]: # ,40,45
#for map in os.listdir("/mnt/raid-cita/mlokken/data/number_density_maps/maglim/"):
    for map in os.listdir("/mnt/raid-cita/mlokken/pkpatch/number_density_maps/fullsky/1pt5E12_to_1E15_msun/"):
        if "_"+str(scale)+"Mpc" in map and map.endswith("a.fits") and "ORIENT" not in map and "all_z" not in map:
            print(map)
            print(os.path.splitext(map)[0][-3:])
            new_ini_file = "theta_of_r_pkfile_restricted_nue_{:s}Mpc.ini".format(str(scale))
            fout = open(new_ini_file, 'w')
            with open("theta_of_r_pkfile_restricted_nue.ini", 'r') as file:
                for line in file:
                    if line.startswith("map ="):
                        line = line.rstrip()
                        line += map + '\n'
                        fout.write(line)
                    elif line.startswith("fwhm ="):
                        line = line.rstrip()
                        line += " {:s}".format(os.path.splitext(map)[0][-3:-1].replace("_",""))+"\n"
                        fout.write(line)
                    elif line.startswith("output ="):
                        line = line.rstrip()
                        line = line.replace("20",str(scale))
                        fout.write(line+"\n")
                        print(line)
                    else:
                        fout.write(line)
                    
            fout.close()
    subprocess.run(args=["/home/mlokken/software/COOP/mapio/GetPeaks",new_ini_file])
