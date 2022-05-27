import numpy as np

def ThetaPhitoDeclRa(theta,phi):
    dec = -1*np.degrees(theta)+90
    ra  = np.degrees(phi)
    ra[ra>180] = -360 + ra[ra>180]
    return dec,ra

def make_regions(ra, dec, nreg, plot=False, plotroot = None, thetaphi=False, mode='ACTxDES'):
    """
    Split into regions
    if thetaphi=True then first two arguments are theta,phi
    """
    import kmeans_radec
    from kmeans_radec import KMeans, kmeans_sample
    import matplotlib.pyplot as plt
    import astropy.units as u
    import astropy.coordinates as coord

    if thetaphi:
        dec,ra = ThetaPhitoDeclRa(theta, phi)

    km = kmeans_sample(np.vstack((ra,dec)).T, nreg, maxiter=100, tol=1.0e-5)
    
    # did we converge?
    print("regions converged?",km.converged)
    
    # how many in each cluster? Should be fairly uniform
    print("cluster sizes:", np.bincount(km.labels))
    
    if plot:
        print("Plotting.")
        if mode == 'ACTxDES':
            plt.figure(figsize=[11,3])
        else:
            plt.figure(figsize=[10,10])
        cmap = plt.cm.get_cmap('prism',60)
        for i in range(nreg):
            in_reg = km.labels == i
            racoord = coord.Angle(ra[in_reg]*u.degree)
            racoord = racoord.wrap_at(180*u.degree)
            dec_coord = coord.Angle(dec[in_reg]*u.degree)
            plt.scatter(racoord,dec_coord,color=cmap(i))
            plt.xlabel("RA")
            plt.ylabel("Dec")
        if plotroot == None:
            print("No filename root provided, displaying plot.")
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig("/home/mlokken/oriented_stacking/plots/paper2/regions_plots/{:s}_regions.png".format(plotroot))
    
    return km.labels

def make_ini_files(pk_mapfile, stk_mapfile, thetaphi_file, rsmooth_Mpc, standard_pk_file, standard_stk_file, outpath, inifile_root, distbin, pk_mask=None, stk_mask=None, e_min=None, e_max=None, nu_min=None, rad_Mpc=None):
    import sys
    import astropy.units as u
    from astropy.cosmology import Planck18 as cosmo
    from astropy.cosmology import z_at_value
    import os
    from pathlib import Path

    ''' 
     
    '''
    orient_mode = 'HESSIAN'

    if ('thetaphi' not in thetaphi_file) or (not thetaphi_file.endswith('txt')):
        sys.exit("Wrong thetaphi file: you have inputted {:s}".format(thetaphi_file))
    lowerlim, upperlim = distbin[0], distbin[1]
    mid = (float(lowerlim) + float(upperlim))/2. * u.megaparsec
    # get the redshift at the middle of this slice
    z = z_at_value(cosmo.comoving_distance, mid)
    smth_scale_arcmin = (cosmo.arcsec_per_kpc_comoving(z).to(u.arcsec/u.megaparsec)*rsmooth_Mpc*u.Mpc).to(u.arcmin)
    if rad_Mpc is not None:
        radius = (cosmo.arcsec_per_kpc_comoving(z).to(u.deg/u.Mpc))*rad_Mpc*u.Mpc # radius in degrees
        res    = int(radius/((0.5*u.arcmin).to(u.deg))) # final stack has 0.5' pixels
    pkfile  = os.path.join(outpath, inifile_root+"_pks.fits")
    stkfile = os.path.join(outpath, inifile_root+"_stk")
    fout_name = os.path.join(outpath,inifile_root+"_pk.ini")
    if fout_name not in os.listdir(outpath):
        Path(fout_name).touch()
    fout = open(fout_name, 'w')

    with open(standard_pk_file, 'r') as file:
        for line in file:
            if line.startswith("map ="):
                fout.write(line.replace(line, "map = {:s}\n".format(pk_mapfile)))
            elif line.startswith("mask ="):
                if pk_mask is not None:
                    fout.write(line.replace(line, "mask = {:s}\n".format(pk_mask)))
            elif line.startswith("output ="):
                fout.write(line.replace(line, "output = {:s}\n".format(pkfile)))
            elif line.startswith("external_list ="):
                fout.write(line.replace(line, "external_list = {:s}\n".format(thetaphi_file)))
            elif line.startswith("orient ="):
                fout.write(line.replace(line, "orient = {:s}".format(orient_mode)))
            elif line.startswith("fwhm_orient"):
                fout.write(line.replace(line, "fwhm_orient = {:s}\n".format(str(int(smth_scale_arcmin.value)))))
            elif line.startswith("e_min ="):
                if e_min is not None:
                    fout.write(line.replace(line, "e_min = {:0.2f}\n".format(e_min)))
            elif line.startswith("e_max ="):
                if e_max is not None:
                    fout.write(line.replace(line, "e_max = {:0.2f}\n".format(e_max)))
            elif line.startswith("fwhm_e ="):
                if (e_min is not None) or (e_max is not None):
                    fout.write(line.replace(line, "fwhm_e = {:s}\n".format(str(int(smth_scale_arcmin.value)))))
            elif line.startswith("nu_min ="):
                if nu_min is not None:
                    fout.write(line.replace(line, "nu_min = {:.2f}\n".format(nu_min)))
            elif line.startswith("fwhm_nu ="):
                if nu_min is not None:
                    fout.write(line.replace(line, "fwhm_nu = {:s}\n".format(str(int(smth_scale_arcmin.value)))))
            elif line.startswith("symmetry ="):
                fout.write(line.replace(line, "symmetry = SYMMETRIC\n"))
            elif line.startswith("sym_option = "):
                fout.write(line.replace(line, "sym_option = {:s}\n".format(orient_mode)))
            elif line.startswith("fwhm_sym ="):
                fout.write(line.replace(line, "fwhm_sym = {:s}\n".format(str(int(smth_scale_arcmin.value)))))
            else:
                fout.write(line)
    fout.close()

    fout_stk_name = os.path.join(outpath,inifile_root+"_stk.ini")
    if fout_stk_name not in os.listdir(outpath):
        Path(fout_stk_name).touch()
    fout_stk = open(fout_stk_name, 'w')
    with open(standard_stk_file, 'r') as file:
        for line in file:
            if line.startswith("map ="):
                fout_stk.write(line.replace(line, "map = {:s}\n".format(stk_mapfile)))
            elif line.startswith("mask ="):
                if stk_mask is not None:
                    fout_stk.write(line.replace(line, "mask = {:s}\n".format(stk_mask)))
            elif line.startswith("peaks = "):
                fout_stk.write(line.replace(line, "peaks = {:s}\n".format(pkfile)))
            elif line.startswith("output = "):
                fout_stk.write(line.replace(line, "output = {:s}\n".format(stkfile)))
            elif (line.startswith("radius =")) and (rad_Mpc is not None):
                fout_stk.write(line.replace(line, "radius = {:.2f}\n".format(radius.value)))
            elif (line.startswith("res =")) and (rad_Mpc is not None):
                fout_stk.write(line.replace(line, "res = {:d}\n".format(res)))
            else:
                fout_stk.write(line)
    fout_stk.close()
    return fout_name, fout_stk_name

def make_pk_ini_file(pk_mapfile, rsmooth_Mpc, standard_pk_file, outpath, inifile_root, distbin, thetaphi_file=None, pk_mask=None, e_min=None, e_max=None, nu_min=None):
    import sys
    import astropy.units as u
    from astropy.cosmology import Planck15 as cosmo
    from astropy.cosmology import z_at_value
    import os
    from pathlib import Path

    orient_mode = 'HESSIAN'

    if thetaphi_file is not None:
        if ('thetaphi' not in thetaphi_file) or (not thetaphi_file.endswith('txt')):
            sys.exit("Wrong thetaphi file: you have inputted {:s}".format(thetaphi_file))
    lowerlim, upperlim = distbin[0], distbin[1]
    mid = (float(lowerlim) + float(upperlim))/2. * u.megaparsec
    # get the redshift at the middle of this slice
    z = z_at_value(cosmo.comoving_distance, mid)
    smth_scale_arcmin = (cosmo.arcsec_per_kpc_comoving(z).to(u.arcsec/u.megaparsec)*rsmooth_Mpc*u.Mpc).to(u.arcmin)
    pkfile  = os.path.join(outpath, inifile_root+"_pks.fits")
    fout_name = os.path.join(outpath,inifile_root+"_pk.ini")
    if fout_name not in os.listdir(outpath):
        Path(fout_name).touch()
    fout = open(fout_name, 'w')

    with open(standard_pk_file, 'r') as file:
        for line in file:
            if line.startswith("map ="):
                fout.write(line.replace(line, "map = {:s}\n".format(pk_mapfile)))
            elif line.startswith("mask ="):
                if pk_mask is not None:
                    fout.write(line.replace(line, "mask = {:s}\n".format(pk_mask)))
                else:
                    fout.write(line)
            elif line.startswith("output ="):
                fout.write(line.replace(line, "output = {:s}\n".format(pkfile)))
            elif line.startswith("external_list ="):
                if thetaphi_file is not None:
                    fout.write(line.replace(line, "external_list = {:s}\n".format(thetaphi_file)))
                else:
                    fout.write("point_type = MAX\nfwhm_pt = {:s}\n".format(str(int(smth_scale_arcmin.value))))
            elif line.startswith("orient ="):
                fout.write(line.replace(line, "orient = {:s}".format(orient_mode)))
            elif line.startswith("fwhm_orient"):
                fout.write(line.replace(line, "fwhm_orient = {:s}\n".format(str(int(smth_scale_arcmin.value)))))
            elif line.startswith("e_min ="):
                if e_min is not None:
                    fout.write(line.replace(line, "e_min = {:0.2f}\n".format(e_min)))
                else:
                    fout.write(line)
            elif line.startswith("e_max ="):
                if e_max is not None:
                    fout.write(line.replace(line, "e_max = 1000\n"))
                else:
                    fout.write(line)
            elif line.startswith("fwhm_e ="):
                if (e_min is not None) or (e_max is not None):
                    fout.write(line.replace(line, "fwhm_e = {:s}\n".format(str(int(smth_scale_arcmin.value)))))
                else:
                    fout.write(line)
            elif line.startswith("nu_min ="):
                if nu_min is not None:
                    fout.write(line.replace(line, "nu_min = {:.2f}\n".format(nu_min)))
                else:
                    fout.write(line)
            elif line.startswith("fwhm_nu ="):
                if nu_min is not None:
                    fout.write(line.replace(line, "fwhm_nu = {:s}\n".format(str(int(smth_scale_arcmin.value)))))
                else:
                    fout.write(line)
            elif line.startswith("symmetry ="):
                fout.write(line.replace(line, "symmetry = SYMMETRIC\n"))
            elif line.startswith("sym_option = "):
                fout.write(line.replace(line, "sym_option = {:s}\n".format(orient_mode)))
            elif line.startswith("fwhm_sym ="):
                fout.write(line.replace(line, "fwhm_sym = {:s}\n".format(str(int(smth_scale_arcmin.value)))))
            else:
                fout.write(line)
    fout.close()

    return fout_name

def make_pk_ini_file_norot(pk_mapfile, standard_pk_file, outpath, inifile_root, rsmooth_Mpc=None, distbin=None, thetaphi_file=None, pk_mask=None, e_min=None, e_max=None, nu_min=None):
    import sys
    import astropy.units as u
    from astropy.cosmology import Planck15 as cosmo
    from astropy.cosmology import z_at_value
    import os
    from pathlib import Path

    orient_mode = 'RANDOM'

    if thetaphi_file is not None:
        if ('thetaphi' not in thetaphi_file) or (not thetaphi_file.endswith('txt')):
            sys.exit("Wrong thetaphi file: you have inputted {:s}".format(thetaphi_file))
    if rsmooth_Mpc is not None:
        lowerlim, upperlim = distbin[0], distbin[1]
        mid = (float(lowerlim) + float(upperlim))/2. * u.megaparsec
        # get the redshift at the middle of this slice
        z = z_at_value(cosmo.comoving_distance, mid)
        smth_scale_arcmin = (cosmo.arcsec_per_kpc_comoving(z).to(u.arcsec/u.megaparsec)*rsmooth_Mpc*u.Mpc).to(u.arcmin)
    pkfile  = os.path.join(outpath, inifile_root+"_pks.fits")
    fout_name = os.path.join(outpath,inifile_root+"_pk.ini")
    if fout_name not in os.listdir(outpath):
        Path(fout_name).touch()
    fout = open(fout_name, 'w')

    with open(standard_pk_file, 'r') as file:
        for line in file:
            if line.startswith("map ="):
                fout.write(line.replace(line, "map = {:s}\n".format(pk_mapfile)))
            elif line.startswith("mask ="):
                if pk_mask is not None:
                    fout.write(line.replace(line, "mask = {:s}\n".format(pk_mask)))
                else:
                    fout.write(line)
            elif line.startswith("output ="):
                fout.write(line.replace(line, "output = {:s}\n".format(pkfile)))
            elif line.startswith("external_list ="):
                if thetaphi_file is not None:
                    fout.write(line.replace(line, "external_list = {:s}\n".format(thetaphi_file)))
                else:
                    fout.write("point_type = MAX\nfwhm_pt = {:s}\n".format(str(int(smth_scale_arcmin.value))))
            elif line.startswith("orient ="):
                fout.write(line.replace(line, "orient = {:s}".format(orient_mode)))
            elif line.startswith("e_min ="):
                if e_min is not None:
                    fout.write(line.replace(line, "e_min = {:0.2f}\n".format(e_min)))
                else:
                    fout.write(line)
            elif line.startswith("e_max ="):
                if e_max is not None:
                    fout.write(line.replace(line, "e_max = 1000\n"))
                else:
                    fout.write(line)
            elif line.startswith("fwhm_e ="):
                if (e_min is not None) or (e_max is not None):
                    fout.write(line.replace(line, "fwhm_e = {:s}\n".format(str(int(smth_scale_arcmin.value)))))
                else:
                    fout.write(line)
            elif line.startswith("nu_min ="):
                if nu_min is not None:
                    fout.write(line.replace(line, "nu_min = {:.2f}\n".format(nu_min)))
                else:
                    fout.write(line)
            elif line.startswith("fwhm_nu ="):
                if nu_min is not None:
                    fout.write(line.replace(line, "fwhm_nu = {:s}\n".format(str(int(smth_scale_arcmin.value)))))
                else:
                    fout.write(line)

            else:
                fout.write(line)
    fout.close()

    return fout_name

def make_stk_ini_file(pkfile, stk_mapfile, standard_stk_file, outpath, inifile_root, distbin,stk_mask=None, rad_Mpc=None):
    import sys
    import astropy.units as u
    from astropy.cosmology import Planck18 as cosmo
    from astropy.cosmology import z_at_value
    import os
    from pathlib import Path

    lowerlim, upperlim = distbin[0], distbin[1]
    mid = (float(lowerlim) + float(upperlim))/2. * u.megaparsec
    # get the redshift at the middle of this slice 
    z = z_at_value(cosmo.comoving_distance, mid)
    if rad_Mpc is not None:
        radius = (cosmo.arcsec_per_kpc_comoving(z).to(u.deg/u.Mpc))*rad_Mpc*u.Mpc # radius in degrees
        res    = int(radius/((0.5*u.arcmin).to(u.deg))) # final stack has 0.5' pixels
    stkfile = os.path.join(outpath, inifile_root+"_stk")

    fout_stk_name = os.path.join(outpath,inifile_root+"_stk.ini")
    if fout_stk_name not in os.listdir(outpath):
        Path(fout_stk_name).touch()
    fout_stk = open(fout_stk_name, 'w')

    with open(standard_stk_file, 'r') as file:
        for line in file:
            if line.startswith("map ="):
                fout_stk.write(line.replace(line, "map = {:s}\n".format(stk_mapfile)))
            elif line.startswith("mask ="):
                if stk_mask is not None:
                    fout_stk.write(line.replace(line, "mask = {:s}\n".format(stk_mask)))
            elif line.startswith("peaks = "):
                fout_stk.write(line.replace(line, "peaks = {:s}\n".format(pkfile)))
            elif line.startswith("output = "):
                fout_stk.write(line.replace(line, "output = {:s}\n".format(stkfile)))
            elif (line.startswith("radius =")) and (rad_Mpc is not None):
                fout_stk.write(line.replace(line, "radius = {:.2f}\n".format(radius.value)))
            elif (line.startswith("res =")) and (rad_Mpc is not None):
                fout_stk.write(line.replace(line, "res = {:d}\n".format(res)))
            else:
                fout_stk.write(line)
    fout_stk.close()
    return fout_stk_name
