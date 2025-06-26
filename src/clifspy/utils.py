import argparse
import glob
import logging
import re

from astropy import coordinates
from astropy.table import Table
from astropy import nddata
from astropy import units
import numpy as np
from photutils import aperture

logger = logging.getLogger("CLIFS_Pipeline")

def r_mask(galaxy, r_as, wcs=None, round=False):
    if round:
        aper_sky = aperture.SkyEllipticalAperture(galaxy.c, r_as*units.arcsec, r_as*units.arcsec)
    else:
        aper_sky = aperture.SkyEllipticalAperture(galaxy.c, r_as*units.arcsec, r_as*units.arcsec*(1-galaxy.ell), theta=galaxy.pa)
    if wcs is None:
        aper_px = aper_sky.to_pixel(galaxy.get_eline_map("Ha-6564", return_map=False, return_wcs=True))
    else:
        aper_px = aper_sky.to_pixel(wcs)
    return aper_px.to_mask(method="center")

def spectral_axis_from_wcs(wcs, nwave, unit="angstrom"):
    coo = np.array([np.ones(nwave), np.ones(nwave), np.arange(nwave) + 1]).T
    wave = wcs.all_pix2world(coo, 1)[:, 2] * wcs.wcs.cunit[2].to(unit)
    return wave

def match_to_galaxy(tcat, ra, dec, max_sep_arcsec=1.0):
    c_cat = coordinates.SkyCoord(tcat["RA"], tcat["DEC"], unit="deg")
    c_gal = coordinates.SkyCoord(ra, dec, unit="deg")
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

class clifs_config_file:
    def __init__(self, clifs_id):
        self.clifs_id = clifs_id
        self._clifstab = Table.read("/arc/projects/CLIFS/catalogs/clifs_master_catalog.fits")
        self.tclifs = self._clifstab[self._clifstab["clifs_id"] == self.clifs_id]

    def _populate_galaxy(self, file):
        print('[galaxy]', file=file)
        print('name = "{}"'.format(self.tclifs["name"][0]), file=file)
        print('clifs_id = {}'.format(self.clifs_id), file=file)
        print('ra = {:.6e}'.format(self.tclifs["ra"][0]), file=file)
        print('dec = {:.5e}'.format(self.tclifs["decl"][0]), file=file)
        print('z = {:.5f}'.format(self.tclifs["redshift"][0]), file=file)
        tnsa = Table.read("/arc/projects/CLIFS/catalogs/nsa_v1_0_1_shrunk.fits")
        pa, ellip, r50, r90 = match_to_galaxy(tnsa, self.tclifs["ra"], self.tclifs["decl"])
        print('ell = {:.3f}'.format(ellip), file=file)
        print('reff = {:.3e}'.format(r50), file=file)
        print('pa = {:.3e}'.format(pa), file=file)
        print('r90 = {:.3e}'.format(r90), file=file)
        print("", file=file)

    def _populate_data_coverage(self, file):
        print('[data_coverage]', file=file)
        if "MaNGA" in self.tclifs["IFU_flag"][0]:
            print('manga = true', file=file)
        else:
            print('manga = false', file=file)
        if "WEAVE" in self.tclifs["IFU_flag"][0]:
            print('weave = true', file=file)
        else:
            print('weave = false', file=file)
        if "ACA" in self.tclifs["CO_flag"][0]:
            print('aca = true', file=file)
        else:
            print('aca = false', file=file)
        if "IRAM" in self.tclifs["CO_flag"][0]:
            print('iram = true', file=file)
        else:
            print('iram = false', file=file)
        print("", file=file)

    def _populate_cube(self, file):
        print('[cube]', file=file)
        print('xmin = -99', file=file)
        print('xmax = -99', file=file)
        print('ymin = -99', file=file)
        print('ymax = -99', file=file)
        print("", file=file)

    def _populate_files(self, file):
        print('[files]', file=file)
        paths = glob.glob("/arc/projects/CLIFS/cubes/clifs/clifs{}/weave/stackcube_???????.fit".format(self.clifs_id))
        if len(paths) == 2:
            numbers = [re.findall(r"\d+", paths[0]), re.findall(r"\d+", paths[1])]
            if int(numbers[0][1]) > int(numbers[1][1]):
                print('cube_blue = "{}"'.format(paths[0]), file=file)
                print('cube_red = "{}"'.format(paths[1]), file=file)
            else:
                print('cube_blue = "{}"'.format(paths[1]), file=file)
                print('cube_red = "{}"'.format(paths[0]), file=file)
            if self.tclifs["weave_obs"][0] == 1:
                print('cube_sci = "/arc/projects/CLIFS/cubes/clifs/clifs{}/weave/calibrated_cube.fits"'.format(self.clifs_id), file=file)
            elif self.tclifs["manga_obs"][0] == 1:
                file_list = glob.glob("/arc/projects/CLIFS/cubes/clifs/clifs{}/manga/*LOGCUBE.fits.gz".format(self.clifs_id))
                print('cube_sci = "{}"'.format(file_list[0]), file=file)
            print('outdir = "/arc/projects/CLIFS/cubes/clifs/clifs{}/weave"'.format(self.clifs_id), file=file)
            print('outdir_dap = "/arc/projects/CLIFS/dap_output/clifs/clifs{}"'.format(self.clifs_id), file=file)
        elif len(paths) == 0:
            logger.info("No 'stackcube' files found")
            print('outdir = "/arc/projects/CLIFS/cubes/clifs/clifs{}/weave"'.format(self.clifs_id), file=file)
            print('outdir_dap = "/arc/projects/CLIFS/dap_output/clifs/clifs{}"'.format(self.clifs_id), file=file)
            if self.tclifs["manga_obs"][0] == 1:
                file_list = glob.glob("/arc/projects/CLIFS/cubes/clifs/clifs{}/manga/*LOGCUBE.fits.gz".format(self.clifs_id))
                print('cube_sci = "{}"'.format(file_list[0]), file=file)
        else:
            raise Exception("Strange number of matches from file search")
        print('outdir_products = "/arc/projects/CLIFS/derived_products/clifs/clifs{}"'.format(self.clifs_id), file=file)
        print("", file=file)

    def _populate_pipeline(self, file):
        print('[pipeline]', file=file)
        print('bkgsub = true', file=file)
        print('bkgsub_galmask = true', file=file)
        print('downsample_spatial = true', file=file)
        print('alpha = 1.28', file=file)
        print('factor_spatial = 2', file=file)
        print('downsample_wav = true', file=file)
        print('fill_ccd_gaps = false', file=file)
        print('fix_astrometry = false', file=file)
        print('hdf5 = true', file=file)
        print('verbose = false', file=file)
        print('clobber = true', file=file)
        print("", file=file)

    def _populate_plotting(self, file):
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

    def make(self):
        outfile = open(f"/arc/projects/CLIFS/config_files/clifs_{self.clifs_id}.toml", "w")
        self._populate_galaxy(outfile)
        self._populate_data_coverage(outfile)
        self._populate_cube(outfile)
        self._populate_files(outfile)
        self._populate_pipeline(outfile)
        self._populate_plotting(outfile)

class field_config_file:
    def __init__(self, plateifu):
        self.plateifu = plateifu
        self._tmanga_full = Table.read("/arc/projects/CLIFS/catalogs/drpall-v3_1_1.fits", hdu=1)
        self.tmanga = self._tmanga_full[self._tmanga_full["plateifu"] == self.plateifu]

    def _populate_galaxy(self, file):
        print('[galaxy]', file=file)
        print('plateifu = "{}"'.format(self.plateifu), file=file)
        print('ra = {:.6e}'.format(self.tmanga["objra"][0]), file=file)
        print('dec = {:.5e}'.format(self.tmanga["objdec"][0]), file=file)
        print('ifura = {:.6e}'.format(self.tmanga["ifura"][0]), file=file)
        print('ifudec = {:.5e}'.format(self.tmanga["ifudec"][0]), file=file)
        print('z = {:.5f}'.format(self.tmanga["z"][0]), file=file)
        tnsa = Table.read("/arc/projects/CLIFS/catalogs/nsa_v1_0_1_shrunk.fits")
        pa, ellip, r50, r90 = match_to_galaxy(tnsa, self.tmanga["objra"], self.tmanga["objdec"])
        print('ell = {:.3f}'.format(ellip), file=file)
        print('reff = {:.3e}'.format(r50), file=file)
        print('pa = {:.3e}'.format(pa), file=file)
        print('r90 = {:.3e}'.format(r90), file=file)
        print("", file=file)

    def _populate_files(self, file):
        print('[files]', file=file)
        print('cube_sci = "/arc/projects/CLIFS/cubes/control/{}/manga-{}-LOGCUBE.fits.gz"'.format(self.plateifu, self.plateifu), file=file)
        print('outdir_dap = "/arc/projects/CLIFS/dap_output/control/{}"'.format(self.plateifu), file=file)
        print('outdir_products = "/arc/projects/CLIFS/derived_products/control/{}"'.format(self.plateifu), file=file)
        print("", file=file)

    def _populate_plotting(self, file):
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

    def make(self):
        outfile = open(f"/arc/projects/CLIFS/config_files/control/control_{self.plateifu}.toml", "w")
        self._populate_galaxy(outfile)
        self._populate_files(outfile)
        self._populate_plotting(outfile)

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
