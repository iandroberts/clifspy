import argparse
import glob
import logging
import re

from astropy import coordinates
from astropy.table import Table
from astropy import nddata
from astropy import units
import numpy as np

logger = logging.getLogger("CLIFS_Pipeline")

def wave_axis_from_wcs(wcs, nwave):
    coo = np.array([np.ones(nwave), np.ones(nwave), np.arange(nwave) + 1]).T
    wave = wcs.all_pix2world(coo, 1)[:, 2] * wcs.wcs.cunit[2].to('angstrom')
    return wave

def match_to_galaxy(tcat, tgal, max_sep_arcsec=1.0):
    c_cat = coordinates.SkyCoord(tcat["RA"], tcat["DEC"], unit="deg")
    c_gal = coordinates.SkyCoord(tgal["ra"], tgal["decl"], unit="deg")
    max_sep = max_sep_arcsec * units.arcsec
    idx, d2d, d3d = c_gal.match_to_catalog_3d(c_cat)
    if d2d[0] < max_sep:
        pa = tcat["ELPETRO_PHI"][idx[0]]
        ellip = 1 - tcat["ELPETRO_BA"][idx[0]]
        r50 = tcat["ELPETRO_TH50_R"][idx[0]]
        r90 = tcat["ELPETRO_TH90_R"][idx[0]]
        return pa, ellip, r50, r90
    else:
        logger.info("No galaxy match in NSA (max_sep = {} arcsec)".format(max_sep_arcsec))
        return -99, -99, -99, -99

def _populate_galaxy(args, file, tclifs):
    print('[galaxy]', file=file)
    print('name = "{}"'.format(tclifs["name"][0]), file=file)
    print('clifs_id = {}'.format(args.clifs_id), file=file)
    print('ra = {:.6e}'.format(tclifs["ra"][0]), file=file)
    print('dec = {:.5e}'.format(tclifs["decl"][0]), file=file)
    print('z = {:.5f}'.format(tclifs["redshift"][0]), file=file)
    tnsa = Table.read("/arc/projects/CLIFS/catalogs/nsa_v1_0_1_shrunk.fits")
    pa, ellip, r50, r90 = match_to_galaxy(tnsa, tclifs)
    print('ell = {:.3f}'.format(ellip), file=file)
    print('reff = {:.3e}'.format(r50), file=file)
    print('pa = {:.3e}'.format(pa), file=file)
    print('r90 = {:.3e}'.format(r90), file=file)
    print("", file=file)

def _populate_data_coverage(args, file, tclifs):
    print('[data_coverage]', file=file)
    if "MaNGA" in tclifs["IFU_flag"][0]:
        print('manga = true', file=file)
    else:
        print('manga = false', file=file)
    if "WEAVE" in tclifs["IFU_flag"][0]:
        print('weave = true', file=file)
    else:
        print('weave = false', file=file)
    if "ACA" in tclifs["CO_flag"][0]:
        print('aca = true', file=file)
    else:
        print('aca = false', file=file)
    if "IRAM" in tclifs["CO_flag"][0]:
        print('iram = true', file=file)
    else:
        print('iram = false', file=file)
    print("", file=file)

def _populate_cube(args, file):
    print('[cube]', file=file)
    print('xmin = {}'.format(args.xmin), file=file)
    print('xmax = {}'.format(args.xmax), file=file)
    print('ymin = {}'.format(args.ymin), file=file)
    print('ymax = {}'.format(args.ymax), file=file)
    print("", file=file)

def _populate_files(args, file, tclifs):
    print('[files]', file=file)
    paths = glob.glob("/arc/projects/CLIFS/cubes/clifs/clifs{}/weave/stackcube_???????.fit".format(args.clifs_id))
    if len(paths) == 2:
        numbers = [re.findall(r"\d+", paths[0]), re.findall(r"\d+", paths[1])]
        if int(numbers[0][1]) > int(numbers[1][1]):
            print('cube_blue = "{}"'.format(paths[0]), file=file)
            print('cube_red = "{}"'.format(paths[1]), file=file)
        else:
            print('cube_blue = "{}"'.format(paths[1]), file=file)
            print('cube_red = "{}"'.format(paths[0]), file=file)
        if tclifs["weave_obs"][0] == 1:
            print('cube_sci = "/arc/projects/CLIFS/cubes/clifs/clifs{}/weave/calibrated_cube.fits"'.format(args.clifs_id), file=file)
        elif tclifs["manga_obs"][0] == 1:
            file_list = glob.glob("/arc/projects/CLIFS/cubes/clifs/clifs{}/manga/*LOGCUBE.fits.gz".format(args.clifs_id))
            print('cube_sci = "{}"'.format(file_list[0]), file=file)
        print('outdir = "/arc/projects/CLIFS/cubes/clifs/clifs{}/weave"'.format(args.clifs_id), file=file)
        print('outdir_dap = "/arc/projects/CLIFS/dap_output/clifs/clifs{}"'.format(args.clifs_id), file=file)
    elif len(paths) == 0:
        logger.info("No 'stackcube' files found")
        print('outdir = "/arc/projects/CLIFS/cubes/clifs/clifs{}/weave"'.format(args.clifs_id), file=file)
        print('outdir_dap = "/arc/projects/CLIFS/dap_output/clifs/clifs{}"'.format(args.clifs_id), file=file)
        if tclifs["manga_obs"][0] == 1:
            file_list = glob.glob("/arc/projects/CLIFS/cubes/clifs/clifs{}/manga/*LOGCUBE.fits.gz".format(args.clifs_id))
            print('cube_sci = "{}"'.format(file_list[0]), file=file)
    else:
        raise Exception("Strange number of matches from file search")
    print("", file=file)

def _populate_pipeline(args, file):
    print('[pipeline]', file=file)
    print('bkgsub = {}'.format(args.bkgsub), file=file)
    print('bkgsub_galmask = {}'.format(args.bkgsub_galmask), file=file)
    print('downsample_spatial = {}'.format(args.downsample_spatial), file=file)
    print('alpha = {}'.format(args.alpha), file=file)
    print('factor_spatial = {}'.format(args.factor_spatial), file=file)
    print('downsample_wav = {}'.format(args.downsample_wav), file=file)
    print('fill_ccd_gaps = {}'.format(args.fill_ccd_gaps), file=file)
    print('fix_astrometry = {}'.format(args.fix_astrometry), file=file)
    print('hdf5 = {}'.format(args.hdf5), file=file)
    print('verbose = {}'.format(args.verbose), file=file)
    print('clobber = {}'.format(args.clobber), file=file)
    print("", file=file)

def _populate_plotting(file):
    print('[plotting]', file=file)
    print('panel.Nr90 = 1.0', file=file)
    print('panel.sn_min = [1, 2]', file=file)
    print('panel.sn_max = [32, 30]', file=file)
    print('panel.v_star_min = [-100, -75]', file=file)
    print('panel.v_star_max = [100, 75]', file=file)
    print('panel.vdisp_star_min = [0, 10]', file=file)
    print('panel.vdisp_star_max = [100, 90]', file=file)
    print('panel.dn4000_min = [1.0, 1.1]', file=file)
    print('panel.dn4000_max = [2.0, 1.9]', file=file)
    print('panel.flux_ha_min = [0, 5]', file=file)
    print('panel.flux_ha_max = [50, 45]', file=file)
    print('panel.v_ha_min = [-100, -75]', file=file)
    print('panel.v_ha_max = [100, 75]', file=file)
    print("", file=file)
    print('fov.b_pct = 99.9', file=file)
    print('fov.g_pct = 99.8', file=file)
    print('fov.r_pct = 99.7', file=file)
    print('fov.asinh_a = 0.05', file=file)
    print("", file=file)
    print('specfit.eline_labels = true', file=file)
    print('specfit.inset_ylim = [1.3, 4.0]', file=file)

def make_config_file(args):
    clifstab = Table.read("/arc/projects/CLIFS/catalogs/clifs_master_catalog.fits")
    tclifs = clifstab[clifstab["clifs_id"] == args.clifs_id]
    file = open(f"/arc/projects/CLIFS/config_files/clifs_{args.clifs_id}.toml", "w")
    _populate_galaxy(args, file, tclifs)
    _populate_data_coverage(args, file, tclifs)
    _populate_cube(args, file)
    _populate_files(args, file, tclifs)
    _populate_pipeline(args, file)
    _populate_plotting(file)

def sky_cutout_from_image(img, coord, size, wcs):
    cut = nddata.Cutout2D(img, coord, size, wcs = wcs)
    return cut.data, cut.wcs.to_header()

def eline_lookup(line):
    # Lookup table to convert line name to correct extension in MaNGA-DAP maps file
    # see: https://sdss-mangadap.readthedocs.io/en/latest/datamodel.html
    if line == "OII-3727":
        return 0
    elif line == "OII-3729":
        return 1
    elif line == "H12-3751":
        return 2
    elif line == "H11-3771":
        return 3
    elif line == "Hthe-3798":
        return 4
    elif line == "Heta-3836":
        return 5
    elif line == "NeIII-3869":
        return 6
    elif line == "HeI-3889":
        return 7
    elif line == "Hzet-3890":
        return 8
    elif line == "NeIII-3968":
        return 9
    elif line == "Heps-3971":
        return 10
    elif line == "Hdel-4102":
        return 11
    elif line == "Hgam-4341":
        return 12
    elif line == "HeII-4687":
        return 13
    elif line == "Hb-4862":
        return 14
    elif line == "OIII-4960":
        return 15
    elif line == "OIII-5008":
        return 16
    elif line == "NI-5199":
        return 17
    elif line == "NI-5201":
        return 18
    elif line == "HeI-5877":
        return 19
    elif line == "OI-6302":
        return 20
    elif line == "OI-6365":
        return 21
    elif line == "NII-6549":
        return 22
    elif line == "Ha-6564":
        return 23
    elif line == "NII-6585":
        return 24
    elif line == "SII-6718":
        return 25
    elif line == "SII-6732":
        return 26
    elif line == "HeI-7067":
        return 27
    elif line == "ArIII-7137":
        return 28
    elif line == "ArIII-7753":
        return 29
    elif line == "Peta-9017":
        return 30
    elif line == "SIII9071":
        return 31
    elif line == "Pzet-9231":
        return 32
    elif line == "SIII-9533":
        return 33
    elif line == "Peps-9548":
        return 34
    else:
        raise ValueError("Invalid line name, see: https://sdss-mangadap.readthedocs.io/en/latest/datamodel.html")

def eline_mask(wave, z, medium="air", dv=500, bright_only=False):
    if medium == "air":
        rest_wav = np.array([3726.032, 3728.815, 3750.158, 3770.637, 3797.904, 3835.391, 3868.760, 3888.647, 3889.064, 3967.470,
                             3970.079, 4101.742, 4340.471, 4685.710, 4861.333, 4958.911, 5006.843, 5197.902, 5200.257, 5875.624,
                             6300.304, 6363.776, 6548.050, 6562.819, 6583.460, 6716.440, 6730.810, 7065.196, 7135.790, 7751.060,
                             9014.909, 9068.600, 9229.014, 9531.100, 9545.969])
        mask = np.ones(wave.size).astype(bool)
        for rw in rest_wav:
            w = rw * (1 + z)
            dl = w * (dv / 2.998e+5)
            mask[(wave > w - dl) & (wave < w + dl)] = False
        return mask
