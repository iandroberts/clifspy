import glob
import sys
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import numpy as np
import toml
from clifspy.utils import eline_lookup
from astropy.table import Table
from astropy import cosmology
from photutils import aperture

class Galaxy:
    def __init__(self, clifs_id):
        clifs_cat = Table.read("/arc/projects/CLIFS/catalogs/clifs_master_catalog.fits")
        manga_cat = Table.read("/arc/projects/CLIFS/catalogs/drpall-v3_1_1.fits", hdu=1)
        config_path = f"/arc/projects/CLIFS/config_files/clifs_{clifs_id}.toml"
        self.config = toml.load(config_path)
        self.name = self.config["galaxy"]["name"]
        self.clifs_id = self.config["galaxy"]["clifs_id"]
        self.ra = self.config["galaxy"]["ra"]
        self.dec = self.config["galaxy"]["dec"]
        self.z = self.config["galaxy"]["z"]
        self.c = SkyCoord(ra = self.ra, dec = self.dec, unit = "deg")
        self.reff = self.config["galaxy"]["reff"] * u.arcsec
        self.ell = self.config["galaxy"]["ell"]
        self.pa = self.config["galaxy"]["pa"] * u.deg
        self.r90 = self.config["galaxy"]["r90"] * u.arcsec
        self.manga = self.config["data_coverage"]["manga"]
        self.weave = self.config["data_coverage"]["weave"]
        self.manga_obs = int(clifs_cat[clifs_cat["clifs_id"] == self.clifs_id]["manga_obs"][0])
        self.weave_obs = int(clifs_cat[clifs_cat["clifs_id"] == self.clifs_id]["weave_obs"][0])
        self.ra_pnt = float(clifs_cat[clifs_cat["clifs_id"] == self.clifs_id]["ra_lifu"][0])
        self.dec_pnt = float(clifs_cat[clifs_cat["clifs_id"] == self.clifs_id]["dec_lifu"][0])
        self.tail = float(clifs_cat[clifs_cat["clifs_id"] == self.clifs_id]["tail_pa"]) >= 0.0
        self.cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
        self.sfr_gswlc = clifs_cat[clifs_cat["clifs_id"] == self.clifs_id]["sfr_gswlc2"][0]
        self.sfr_gswlc_err = clifs_cat[clifs_cat["clifs_id"] == self.clifs_id]["sfr_e_gswlc2"][0]
        if self.manga:
            self.plateifu = clifs_cat[clifs_cat["clifs_id"] == self.clifs_id]["plateifu"][0]
            self.manga_Nfib = int(manga_cat[manga_cat["plateifu"] == self.plateifu]["ifudesignsize"])
    def get_cutout_image(self, telescope, filter, header = False):
        img_path = "/arc/projects/CLIFS/multiwav/cutouts/clifs{}/{}-{}.fits".format(self.clifs_id, telescope, filter)
        return fits.getdata(img_path, header = header)

    def get_maps(self):
        dap_dir = self.config["files"]["outdir_dap"]
        find_maps = glob.glob(dap_dir + "/*-MAPS-HYB10-MILESHC-MASTARSSP.fits*")
        if len(find_maps) == 0:
            return None
        if len(find_maps) == 1:
            mapsfile = fits.open(find_maps[0])
        elif len(find_maps) == 2:
            print("Found two MAPS files, assuming WEAVE and MaNGA, proceeding with WEAVE")
            mapsfile = fits.open([s for s in find_maps if "weave" in s][0])
        else:
            print("Found more than two MAPS files")
            sys.exit()
        return mapsfile

    def get_cube(self, return_ivar=False):
        hdul = fits.open(self.config["files"]["cube_sci"])
        if return_ivar:
            return hdul["FLUX"].data, hdul["IVAR"].data
        return hdul["FLUX"].data

    def get_modelcube(self):
        find_cube = glob.glob(self.config["files"]["outdir_dap"] + "/*-LOGCUBE-HYB10-MILESHC-MASTARSSP.fits*")
        if len(find_cube) == 0:
            return None
        if len(find_cube) == 1:
            cubefile = fits.open(find_cube[0])
        elif len(find_cube) == 2:
            print("Found two LOGCUBE files, assuming WEAVE and MaNGA, proceeding with WEAVE")
            cubefile = fits.open([s for s in find_cube if "weave" in s][0])
        else:
            print("Found more than two MAPS files")
            sys.exit()
        return cubefile

    def get_eline_map(self, line, map = "GFLUX", return_map = True, return_wcs = False):
        mapsfile = self.get_maps()
        if return_wcs and return_map:
            return mapsfile["EMLINE_{}".format(map)].data[eline_lookup(line)], WCS(mapsfile["EMLINE_GFLUX"].header).celestial
        elif return_map:
            return mapsfile["EMLINE_{}".format(map)].data[eline_lookup(line)]
        else:
            return WCS(mapsfile["EMLINE_GFLUX"].header).celestial

    def get_spectrum(self, x, y, only_wave=False):
        data_cube = fits.open(self.config["files"]["cube_sci"])
        flux = data_cube["FLUX"].data
        ivar = data_cube["IVAR"].data
        wcs = WCS(data_cube["FLUX"].header)
        nwave = flux.shape[0]
        coo = np.array([np.ones(nwave), np.ones(nwave), np.arange(nwave) + 1]).T
        wave = (wcs.all_pix2world(coo, 1)[:,2] * wcs.wcs.cunit[2].to("angstrom")) * u.AA
        if only_wave:
            return wave
        x, y = np.round(wcs.celestial.world_to_pixel(self.c)).astype(int)
        data_cube.close()
        return wave.value, flux[:, y, x], ivar[:, y, x]

    def get_sfr_map(self, return_header=False):
        return fits.getdata(self.config["files"]["outdir_products"] + "/sigma_sfr_ha.fits", header=return_header)

    def get_ifu_total_sfr(self, calculate_error=False, Nboot=1000):
        sigsfr, sigsfr_h = self.get_sfr_map(return_header=True)
        kpc2_per_pixel = (self.cosmo.kpc_proper_per_arcmin(self.z).value * sigsfr_h["PC2_2"] * 60) ** 2
        sfr = sigsfr * kpc2_per_pixel
        aper_sky = aperture.SkyEllipticalAperture(self.c, 1.5*self.reff, 1.5*self.reff*(1-self.ell), theta=self.pa)
        aper_px = aper_sky.to_pixel(WCS(sigsfr_h))
        aper_mask = aper_px.to_mask()
        sfr_masked = aper_mask.multiply(sfr)
        sfr_flat = sfr_masked[sfr_masked > 0]
        if calculate_error:
            sfr_boot = np.zeros(Nboot)
            for n in range(Nboot):
                ind = np.random.choice(np.arange(sfr_flat.size), size=sfr_flat.size, replace=True)
                sfr_boot[n] = np.log10(np.nansum(sfr_flat[ind]))
            return np.log10(np.nansum(sfr_flat)), np.nanstd(sfr_boot)
        return np.log10(np.nansum(sfr_masked))

class ControlGalaxy:
    def __init__(self, plateifu):
        self.plateifu = plateifu
        control_cat = Table.read("/arc/projects/CLIFS/catalogs/manga_control_v2.fits")
        manga_cat = Table.read("/arc/projects/CLIFS/catalogs/drpall-v3_1_1.fits", hdu=1)
        config_path = f"/arc/projects/CLIFS/config_files/control/control_{self.plateifu}.toml"
        self.config = toml.load(config_path)
        self.ra = self.config["galaxy"]["ra"]
        self.dec = self.config["galaxy"]["dec"]
        self.z = self.config["galaxy"]["z"]
        self.c = SkyCoord(ra = self.ra, dec = self.dec, unit = "deg")
        self.reff = self.config["galaxy"]["reff"] * u.arcsec
        self.ell = self.config["galaxy"]["ell"]
        self.pa = self.config["galaxy"]["pa"] * u.deg
        self.r90 = self.config["galaxy"]["r90"] * u.arcsec
        self.ra_pnt = self.config["galaxy"]["ifura"]
        self.dec_pnt = self.config["galaxy"]["ifudec"]
        self.cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
        self.sfr_gswlc = control_cat[control_cat["plateifu"] == self.plateifu]["sfr_gswlc2"][0]
        self.sfr_gswlc_err = control_cat[control_cat["plateifu"] == self.plateifu]["sfr_e_gswlc2"][0]
        self.manga_Nfib = int(manga_cat[manga_cat["plateifu"] == self.plateifu]["ifudesignsize"])

    def get_maps(self):
        dap_dir = self.config["files"]["outdir_dap"]
        find_maps = glob.glob(dap_dir + "/*-MAPS-HYB10-MILESHC-MASTARSSP.fits*")
        if len(find_maps) == 0:
            return None
        if len(find_maps) == 1:
            mapsfile = fits.open(find_maps[0])
        else:
            print("Found more than one MAPS files")
            sys.exit()
        return mapsfile

    def get_modelcube(self):
        find_cube = glob.glob(self.config["files"]["outdir_dap"] + "/*-LOGCUBE-HYB10-MILESHC-MASTARSSP.fits*")
        if len(find_cube) == 0:
            return None
        if len(find_cube) == 1:
            cubefile = fits.open(find_cube[0])
        else:
            print("Found more than one LOGCUBE files")
            sys.exit()
        return cubefile

    def get_eline_map(self, line, map = "GFLUX", return_map = True, return_wcs = False):
        mapsfile = self.get_maps()
        if return_wcs and return_map:
            return mapsfile["EMLINE_{}".format(map)].data[eline_lookup(line)], WCS(mapsfile["EMLINE_GFLUX"].header).celestial
        elif return_map:
            return mapsfile["EMLINE_{}".format(map)].data[eline_lookup(line)]
        else:
            return WCS(mapsfile["EMLINE_GFLUX"].header).celestial

    def get_spectrum(self, x, y):
        data_cube = fits.open(self.config["files"]["cube_sci"])
        flux = data_cube["FLUX"].data
        ivar = data_cube["IVAR"].data
        wcs = WCS(data_cube["FLUX"].header)
        nwave = flux.shape[0]
        coo = np.array([np.ones(nwave), np.ones(nwave), np.arange(nwave) + 1]).T
        wave = (wcs.all_pix2world(coo, 1)[:,2] * wcs.wcs.cunit[2].to("angstrom")) * u.AA
        x, y = np.round(wcs.celestial.world_to_pixel(self.c)).astype(int)
        data_cube.close()
        return wave.value, flux[:, y, x], ivar[:, y, x]

    def get_sfr_map(self, return_header=False):
        return fits.getdata(self.config["files"]["outdir_products"] + "/sigma_sfr_ha.fits", header=return_header)

    def get_ifu_total_sfr(self, calculate_error=False, Nboot=1000):
        sigsfr, sigsfr_h = self.get_sfr_map(return_header=True)
        kpc2_per_pixel = (self.cosmo.kpc_proper_per_arcmin(self.z).value * sigsfr_h["PC2_2"] * 60) ** 2
        sfr = sigsfr * kpc2_per_pixel
        aper_sky = aperture.SkyEllipticalAperture(self.c, 1.5*self.reff, 1.5*self.reff*(1-self.ell), theta=self.pa)
        aper_px = aper_sky.to_pixel(WCS(sigsfr_h))
        aper_mask = aper_px.to_mask()
        sfr_masked = aper_mask.multiply(sfr)
        sfr_flat = sfr_masked[sfr_masked > 0]
        if calculate_error:
            sfr_boot = np.zeros(Nboot)
            for n in range(Nboot):
                ind = np.random.choice(np.arange(sfr_flat.size), size=sfr_flat.size, replace=True)
                sfr_boot[n] = np.log10(np.nansum(sfr_flat[ind]))
            return np.log10(np.nansum(sfr_flat)), np.nanstd(sfr_boot)
        return np.log10(np.nansum(sfr_masked))
