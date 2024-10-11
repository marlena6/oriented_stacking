import healpy as hp
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

nmaps = 24
nside = 4096

cls = np.loadtxt("/mnt/raid-cita/mlokken/buzzard/cls_buzzard_for_healpy.txt")
gg1 = cls[:,1]
gg2 = cls[:,2]
gg3 = cls[:,3]
gy1 = cls[:,4]
gy2 = cls[:,5]
gy3 = cls[:,6]
yy  = cls[:,7]

if rank == size-1:
    extras = nmaps % size
else:
    extras = 0
nruns_local = nmaps // size

for i in range(nruns_local+extras):
    d = rank*nruns_local + i
    print(d)
    yfield1, galfield1 = hp.synfast((yy, gg1, gy1), nside=nside, new=True, pol=False)
    hp.write_map("/mnt/raid-cita/mlokken/grf/grf_yfield_buzzardspec_zbin1_{:d}.fits".format(d), yfield1, overwrite=True)
    hp.write_map("/mnt/raid-cita/mlokken/grf/grf_gfield_buzzardspec_zbin1_{:d}.fits".format(d), galfield1, overwrite=True)
    yfield2, galfield2 = hp.synfast((yy, gg2, gy2), nside=nside, new=True, pol=False)
    hp.write_map("/mnt/raid-cita/mlokken/grf/grf_yfield_buzzardspec_zbin2_{:d}.fits".format(d), yfield2, overwrite=True)
    hp.write_map("/mnt/raid-cita/mlokken/grf/grf_gfield_buzzardspec_zbin2_{:d}.fits".format(d), galfield2, overwrite=True)
    yfield3, galfield3 = hp.synfast((yy, gg3, gy3), nside=nside, new=True, pol=False)
    hp.write_map("/mnt/raid-cita/mlokken/grf/grf_yfield_buzzardspec_zbin3_{:d}.fits".format(d), yfield3, overwrite=True)
    hp.write_map("/mnt/raid-cita/mlokken/grf/grf_gfield_buzzardspec_zbin3_{:d}.fits".format(d), galfield3, overwrite=True)
