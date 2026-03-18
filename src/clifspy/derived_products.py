from importlib.resources import as_file, files
from pathlib import Path
import sys

from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, SigmaClip
import astropy.units as u
from astropy.wcs import WCS
import bagpipes as pipes
import numpy as np
from photutils import aperture
from photutils.background import MMMBackground
from reproject import reproject_exact
from scipy.optimize import curve_fit
from scipy.stats import norm

from clifspy.utils import eline_lookup, eline_mask, mjysr_to_px, filter_pivot
import clifspy.galaxy

def colour_excess(wav):
    return (-4.61777 + 1.41612 * np.power(wav.to(u.micron).value, -1) + 1.52077 * np.power(wav.to(u.micron).value, -2) -
    0.63269 * np.power(wav.to(u.micron).value, -3) + 0.07386 * np.power(wav.to(u.micron).value, -4))

def bpt_sf_mask(hb, oiii, ha, nii, divide="kauffmann03"):
    if divide == "kauffmann03":
        dividing_line = 0.61 / (np.log10(nii/ha) - 0.05) + 1.3
    elif divide == "kewley01":
        dividing_line = 0.61 / (np.log10(nii/ha) - 0.47) + 1.19
    else:
        raise ValueError("Invalid BPT classification")
    return np.less_equal(np.log10(oiii/hb), dividing_line)

def dered_ha_flux(galaxy, return_fha=False, sncut=3):
    ha_flux, wcs = galaxy.get_single_map("Ha-6564", return_wcs=True)
    ha_snr = ha_flux * np.sqrt(galaxy.get_single_map("Ha-6564", map = "GFLUX_IVAR"))
    hb_flux = galaxy.get_single_map("Hb-4862")
    hb_snr = hb_flux * np.sqrt(galaxy.get_single_map("Hb-4862", map = "GFLUX_IVAR"))
    oiii_flux = galaxy.get_single_map("OIII-5008")
    oiii_snr = oiii_flux * np.sqrt(galaxy.get_single_map("OIII-5008", map = "GFLUX_IVAR"))
    nii_flux = galaxy.get_single_map("NII-6585")
    nii_snr = nii_flux * np.sqrt(galaxy.get_single_map("NII-6585", map = "GFLUX_IVAR"))
    mask_sn = (np.greater_equal(ha_snr, sncut) & np.greater_equal(hb_snr, sncut)
        & np.greater_equal(oiii_snr, sncut) & np.greater_equal(nii_snr, sncut))
    hb_flux[~mask_sn] = np.nan
    ha_flux[~mask_sn] = np.nan
    oiii_flux[~mask_sn] = np.nan
    nii_flux[~mask_sn] = np.nan
    mask_sf = bpt_sf_mask(hb_flux, oiii_flux, ha_flux, nii_flux, divide="kewley01")
    if return_fha:
        if galaxy.reff.value == -99:
            return -99.0
        aper_sky = aperture.SkyEllipticalAperture(galaxy.c, 1.5*galaxy.reff, 1.5*galaxy.reff*(1-galaxy.ell), theta=galaxy.pa)
        aper_px = aper_sky.to_pixel(wcs)
        aper_mask = aper_px.to_mask()
        ha_masked = aper_mask.multiply(ha_flux)
        ha_tot = np.nansum(ha_flux)
    ha_flux[~mask_sf] = np.nan
    if return_fha:
        ha_masked = aper_mask.multiply(ha_flux)
        fha = np.nansum(ha_masked) / ha_tot
        if np.isnan(fha):
            return -1.0
        return fha
    decr = ha_flux / hb_flux
    return 1e-17 * ha_flux * np.power(decr / 2.85, 0.76 * (colour_excess(656.46 * u.nm) + 4.5)) * (u.erg / u.s / u.cm ** 2)

def make_sfr_map(galaxy, cdelt = 1 / 3600, C = 41.27, H0 = 70, Om0 = 0.3):
    # Defaults to calibration from Kennicutt & Evans, but can set your preferred C if needed (log SFR = log L - log C)
    cosmo = FlatLambdaCDM(H0 = H0, Om0 = Om0)
    Dl = cosmo.luminosity_distance(galaxy.zcoma)
    scale = cosmo.kpc_proper_per_arcmin(galaxy.zcoma).value * 60
    flux = dered_ha_flux(galaxy, sncut=2)
    lum = (4 * np.pi * Dl ** 2 * flux).cgs.value
    sfr = lum / 10 ** C
    sfr[~np.isfinite(sfr)] = np.nan
    Apx_kpc = (cdelt * scale) ** 2
    #if galaxy.ell >= 0:
    #    sigSFR = (sfr / Apx_kpc) * (1 - galaxy.ell)
    #else:
    #    sigSFR = sfr / Apx_kpc
    return sfr / Apx_kpc

def write_sfr_map(galaxy, sig_sfr):
    wcs = galaxy.get_single_map("Ha-6564", return_map = False, return_wcs = True)
    hdr = wcs.to_header()
    hdr["BUNIT"] = ("Msun/yr/kpc2", "Unit of the map")
    hdu = fits.PrimaryHDU(data = sig_sfr, header = hdr)
    Path(galaxy.config["files"]["outdir_products"]).mkdir(parents=True, exist_ok=True)
    hdu.writeto(galaxy.config["files"]["outdir_products"] + "/sigma_sfr_ha.fits", overwrite=True)

def filter_image(galaxy, fltr, return_hdr=False, write=False):
    with as_file(files("clifspy.FILTERS").joinpath(f"{fltr}.dat")) as path:
        lam, resp = np.loadtxt(path, unpack=True)
    flux, hdr = galaxy.get_cube(return_hdr=True)
    wcs = WCS(hdr)
    nwave = flux.shape[0]
    wave = wcs.spectral.pixel_to_world(np.arange(nwave)).to(u.angstrom).value
    resp_interp = np.interp(wave, lam, resp, left=0, right=0)
    resp_3d = resp_interp[:, np.newaxis, np.newaxis]
    f_band = np.trapz(flux * resp_3d, wave, axis=0) / np.trapz(resp_interp, wave)
    stats = sigma_clipped_stats(f_band, maxiters=None)
    bkg_value = 2.5 * stats[1] - 1.5 * stats[0]
    f_band += bkg_value
    if write:
        hdu = fits.PrimaryHDU(data=f_band, header=wcs.celestial.to_header())
        hdu.writeto(galaxy.config["files"]["outdir_products"] + f"/{fltr}.fits",
            overwrite=True)
    if return_hdr:
        return f_band, wcs.celestial.to_header()
    return f_band

def filter_flux_sun(fltr):
    with as_file(files("clifspy.FILTERS").joinpath(f"{fltr}.dat")) as path:
        lam, resp = np.loadtxt(path, unpack=True)
    with as_file(files("clifspy.FILTERS").joinpath(f"solarspectra.dat")) as path:
        wave, flux = np.loadtxt(path, unpack=True)
    resp_interp = np.interp(wave, lam, resp, left=0, right=0)
    f_band = np.trapz(flux * resp_interp, wave) / np.trapz(resp_interp, wave)
    return f_band

def make_mstar_map(galaxy, a=-0.831, b=0.979, H0=70, Om0=0.3):
    # Roediger & Corteau (2015)
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    Dl = cosmo.luminosity_distance(galaxy.zcoma)
    scale = cosmo.kpc_proper_per_arcmin(galaxy.zcoma).value * 60
    dir = "/arc/projects/CLIFS/multiwav/cutouts/clifs{}".format(galaxy.clifs_id)
    fpath = Path(f"{dir}/cfht-G.fits")
    if fpath.is_file():
        fi, hdr = galaxy.get_cutout_image("cfht", "I2", header=True)
        fg = galaxy.get_cutout_image("cfht", "G")
        fi_sun = filter_flux_sun("CFHT.I2")
        piv = filter_pivot("CFHT.I2")
    else:
        fi, hdr = galaxy.get_cutout_image("legacy", "i", header=True)
        fg = galaxy.get_cutout_image("legacy", "g")
        fi_sun = filter_flux_sun("legacy.i")
        piv = filter_pivot("legacy.i")
    wcs_weave = galaxy.get_single_map("Ha-6564", return_map=False, return_wcs=True)
    hdr_weave = wcs_weave.to_header()
    cdelt = hdr_weave["CDELT2"]
    fi = reproject_exact((fi, hdr), wcs_weave, shape_out=wcs_weave.array_shape,
        return_footprint=False)
    fg = reproject_exact((fg, hdr), wcs_weave, shape_out=wcs_weave.array_shape,
        return_footprint=False)
    fi_sun = 3.34e+4 * piv**2 * fi_sun # convert to Jy
    fi = 1e+6*mjysr_to_px(fi, cdelt)
    fg = 1e+6*mjysr_to_px(fg, cdelt)
    MtoL = 10**a * (fg / fi)**(-2.5*b)
    Li = 4 * np.pi * Dl**2 * (fi * u.Jy)
    Li_sun = 4 * np.pi * (1*u.AU)**2 * (fi_sun * u.Jy)
    M = MtoL * (Li.cgs.value / Li_sun.cgs.value)
    #mi = -2.5 * np.log10(fi) + 8.9
    #mg = -2.5 * np.log10(fg) + 8.9
    #Mi = mi - 5 * np.log10(Dl.to(u.pc).value)
    #logM = 1.15 * 0.7 * (mg - mi) - 0.4*Mi
    #M = 10**logM
    Apx_pc = (cdelt * scale * 1000) ** 2
    #if galaxy.ell >= 0:
    #    sigM = (M / Apx_pc) * (1 - galaxy.ell)
    #else:
    #    sigM = M / Apx_pc
    return M / Apx_pc

def write_mstar_map(galaxy, sig_M):
    wcs = galaxy.get_single_map("Ha-6564", return_map = False, return_wcs = True)
    hdr = wcs.to_header()
    hdr["BUNIT"] = ("Msun/pc2", "Unit of the map")
    hdu = fits.PrimaryHDU(data = sig_M, header = hdr)
    Path(galaxy.config["files"]["outdir_products"]).mkdir(parents=True, exist_ok=True)
    hdu.writeto(galaxy.config["files"]["outdir_products"] + "/sigma_mstar_gi.fits", overwrite=True)

def load_spectrum_for_bagpipes(ID):
    cid, x, y = ID.split("_")
    this_gal = clifspy.galaxy.galaxy(int(cid))
    wave, flux, ivar = this_gal.get_spectrum(int(x), int(y))
    flux_unc = 1 / np.sqrt(ivar)
    spectrum = np.c_[wave, 1e-17 * flux, 1e-17 * flux_unc]
    mask_sky = (spectrum[:, 0] < 5570.) | (spectrum[:, 0] > 5586.) # mask strong sky line residuals
    mask_eline = eline_mask(wave, this_gal.z)
    mask_good = np.isfinite(flux) & np.isfinite(flux_unc)
    mask_total =  mask_sky & mask_good & mask_eline
    return spectrum[mask_total]

def run_bagpipes(galaxy, x, y):
    agebins = [0., 100., 500.]
    agebins.extend(list(np.geomspace(1000, 13000, 8)))

    continuity = {}
    continuity["massformed"] = (7., 12.)
    continuity["metallicity"] = (0.01, 2.)
    continuity["metallicity_prior"] = "log_10"
    continuity["bin_edges"] = agebins
    for i in range(1, len(continuity["bin_edges"]) - 1):
        continuity["dsfr" + str(i)] = (-8., 8.)
        continuity["dsfr" + str(i) + "_prior"] = "student_t"

    #nebular = {}
    #nebular["logU"] = (-4., -2.)

    dust = {}
    dust["type"] = "Calzetti"
    dust["eta"] = 2.
    dust["Av"] = (0., 3.0)

    fit_instructions = {}
    fit_instructions["redshift"] = (galaxy.z - 500/3e+5, galaxy.z + 500/3e+5)
    fit_instructions["t_bc"] = 0.01
    fit_instructions["continuity"] = continuity
    #fit_instructions["nebular"] = nebular
    fit_instructions["dust"] = dust
    fit_instructions["veldisp"] = (10., 500.)   #km/s
    fit_instructions["veldisp_prior"] = "log_10"

    noise = {}
    noise["type"] = "white_scaled"
    noise["scaling"] = (1., 10.)
    noise["scaling_prior"] = "log_10"
    fit_instructions["noise"] = noise

    pipes_galaxy = pipes.galaxy("{}_{}_{}".format(galaxy.clifs_id, x, y), load_spectrum_for_bagpipes, photometry_exists = False)
    fit = pipes.fit(pipes_galaxy, fit_instructions, run = "spectroscopy")
    fit.fit(verbose = True, n_live = 1000, sampler = "nautilus", pool = 16)
    fit.plot_spectrum_posterior()  # Shows the input and fitted spectrum/photometry
    fit.plot_sfh_posterior()       # Shows the fitted star-formation history
    #fit.plot_1d_posterior()        # Shows 1d posterior probability distributions
    fit.plot_corner()

def products_for_clifspipe(galaxy):
    sig_sfr = make_sfr_map(galaxy)
    write_sfr_map(galaxy, sig_sfr)
    sig_M = make_mstar_map(galaxy)
    write_mstar_map(galaxy, sig_M)
