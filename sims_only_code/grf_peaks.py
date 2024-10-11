import numpy as np
from astropy.cosmology import Planck15 as cosmo
from astropy.io import fits
import subprocess
import os
import coop_setup_funcs as csf
import error_analysis_funcs as ef
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

zbins =[[0.15, 0.35], [0.35, 0.5], [0.5, 0.65]]
dbins = [[cosmo.comoving_distance(z[0]).value,cosmo.comoving_distance(z[1]).value] for z in zbins]
cutmin = 10
cut    = 'lambda'
cutstr = '{:s}gt{:d}'.format(cut,cutmin)
nmaps  = 24

mode = 'allpks'
# mode = 'nu_e_cuts'

buzzard_path  = "/mnt/raid-cita/mlokken/buzzard/catalogs/small_region_buzzard_1.9.9_3y3a_rsshift_run_redmapper_v0.5.1_lgt05_vl50_catalog.fit"
grf_path = "/mnt/scratch-lustre/mlokken/stacking/GRF_buzzspec/"
pkmask = "/mnt/raid-cita/mlokken/masks/y3_gold_2.2.1_wide_sofcol_run_redmapper_v6.4.22_redmagic_highdens_0.5_vlim_zmask_hpx.fit"

ra,dec,z,richness = csf.get_radeczlambda(buzzard_path)
# limit with richness                                                                                                                         
print("Cutting by richness.")
rich_cond = richness > cutmin
ra,dec,z  = ra[rich_cond], dec[rich_cond], z[rich_cond]

nbuzz_thr = [262,1051,1366]

if rank == size-1:
    extras = nmaps % size
else:
    extras = 0
nruns_local = nmaps // size

for i in range(nruns_local+extras):
    d = rank*nruns_local + i
    path = os.path.join(grf_path, "{:d}".format(d))
    if not os.path.exists(path):
        os.mkdir(path)

c = 0
for zbin in zbins:
    if c==0:
        smth = 1.8
        nu_min=2.75
    if c==1:
        smth = 1.6
        nu_min=2.55
    if c==2:
        smth = 1.4
        nu_min=2.55
    nustr = ("{:.1f}".format(nu_min)).replace('.','pt')
    smth_str = ("{:.1f}".format(smth)).replace('.','pt')
    print(zbin[0])
    in_bin = (z>zbin[0]) & (z<zbin[1])
    dlow = int(dbins[c][0])
    dhi  = int(dbins[c][1])
    npks_buzz = len(ra[in_bin])
    print("{:d} Buzzard clusters with lambda>10 in bin {:d}.\n".format(npks_buzz, c))
    for i in range(nruns_local+extras):
        d = rank*nruns_local + i
        grf_map = "/mnt/raid-cita/mlokken/GRF_buzzspec/grf_gfield_buzzardspec_zbin{:d}_{:d}.fits".format(c+1,d)
        inifile_root = "{:s}_{:s}_reg{:d}_{:d}_{:d}Mpc_{:s}_{:s}".format("GRF", cutstr, d, dlow, dhi, "nugt{:s}".format(nustr), smth_str)
        pk_ini = ef.make_pk_ini_file_norot(grf_map, "/home/mlokken/oriented_stacking/general_code/standard_pkfile.ini", os.path.join(grf_path, "{:d}".format(d)), inifile_root, rsmooth_Mpc=smth, distbin=[dlow,dhi], nu_min=nu_min,  pk_mask=pkmask, e_min=None, e_max=None)
        subprocess.run(args=["/home/mlokken/software/COOP/mapio/GetPeaks",pk_ini])
        pkfile = fits.open(os.path.join(grf_path, "{:d}".format(d), inifile_root+"_pks.fits"))
        pkinfo = pkfile[0].data
        pkfile.close()
        theta,phi = pkinfo[:,1],pkinfo[:,2]
        all_pks  = np.arange(len(theta))
        if mode == 'allpks':
            # selected = np.random.choice(all_pks, size=nbuzz_thr[c], replace=False)
            selected = all_pks
            sizestr = ''
        elif mode == 'nu_e_cuts':
            selected = np.random.choice(all_pks, size=int(nbuzz_thr[c]/0.017), replace=False)
            sizestr  = 'lrg'
        print("amount larger:", len(selected)/npks_buzz)
        save_tp  = np.hstack((np.reshape(theta[selected],(len(theta[selected]),1)), np.reshape(phi[selected],(len(phi[selected]),1))))
        np.savetxt(grf_path+"/{:d}/".format(d)+"thetaphi_{:s}_nugt{:s}_reg{:d}_{:d}_{:d}Mpc_{:s}{:s}.txt".format(smth_str,nustr,d,dlow,dhi,cutstr,sizestr), save_tp)
    c+=1

