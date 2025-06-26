import clifspy.galaxy
import bagpipes as pipes
from astropy.io import fits
from clifspy.utils import eline_lookup, eline_mask
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from pathlib import Path
from photutils import aperture

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

def dered_ha_flux(galaxy, return_fha=False):
    ha_flux, wcs = galaxy.get_eline_map("Ha-6564", return_wcs=True)
    ha_snr = ha_flux * np.sqrt(galaxy.get_eline_map("Ha-6564", map = "GFLUX_IVAR"))
    hb_flux = galaxy.get_eline_map("Hb-4862")
    hb_snr = hb_flux * np.sqrt(galaxy.get_eline_map("Hb-4862", map = "GFLUX_IVAR"))
    oiii_flux = galaxy.get_eline_map("OIII-5008")
    oiii_snr = oiii_flux * np.sqrt(galaxy.get_eline_map("OIII-5008", map = "GFLUX_IVAR"))
    nii_flux = galaxy.get_eline_map("NII-6585")
    nii_snr = nii_flux * np.sqrt(galaxy.get_eline_map("NII-6585", map = "GFLUX_IVAR"))
    mask_sn = np.greater_equal(ha_snr, 3) & np.greater_equal(hb_snr, 3) & np.greater_equal(oiii_snr, 3) & np.greater_equal(nii_snr, 3)
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
    # TO-DO: add BPT selection for SFing spaxels
    cosmo = FlatLambdaCDM(H0 = H0, Om0 = Om0)
    Dl = cosmo.luminosity_distance(galaxy.z)
    scale = cosmo.kpc_proper_per_arcmin(galaxy.z).value * 60
    flux = dered_ha_flux(galaxy)
    lum = (4 * np.pi * Dl ** 2 * flux).cgs.value
    sfr = lum / 10 ** C
    sfr[~np.isfinite(sfr)] = np.nan
    Apx_kpc = (cdelt * scale) ** 2
    return sfr / Apx_kpc

def write_sfr_map(galaxy, sig_sfr):
    wcs = galaxy.get_eline_map("Ha-6564", return_map = False, return_wcs = True)
    hdr = wcs.to_header()
    hdr["BUNIT"] = ("Msun/yr/kpc2", "Unit of the map")
    hdu = fits.PrimaryHDU(data = sig_sfr, header = hdr)
    Path(galaxy.config["files"]["outdir_products"]).mkdir(parents=True, exist_ok=True)
    hdu.writeto(galaxy.config["files"]["outdir_products"] + "/sigma_sfr_ha.fits", overwrite=True)

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
    #run_bagpipes(galaxy, 66, 40)
