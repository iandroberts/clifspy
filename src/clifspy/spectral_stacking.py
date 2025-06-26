from photutils import aperture

from clifspy import utils

import pathlib
import time

from IPython import embed

import numpy as np
import numpy.ma as ma

from astropy.io import fits
import astropy.constants
from matplotlib import pyplot
from astropy.table import Table
import argparse
from astropy import units

from mangadap.datacube import MaNGADataCube
from mangadap.config import manga
from mangadap.config import defaults
from mangadap.util.sampling import spectral_coordinate_step, Resample

from mangadap.util.resolution import SpectralResolution
from mangadap.util.pixelmask import SpectralPixelMask

from mangadap.par.artifactdb import ArtifactDB
from mangadap.par.emissionmomentsdb import EmissionMomentsDB
from mangadap.par.emissionlinedb import EmissionLineDB
from mangadap.par.absorptionindexdb import AbsorptionIndexDB
from mangadap.par.bandheadindexdb import BandheadIndexDB

from mangadap.proc.templatelibrary import TemplateLibrary
from mangadap.proc.emissionlinemoments import EmissionLineMoments
from mangadap.proc.sasuke import Sasuke
from mangadap.proc.ppxffit import PPXFFit
from mangadap.proc.stellarcontinuummodel import StellarContinuumModel, StellarContinuumModelBitMask
from mangadap.proc.emissionlinemodel import EmissionLineModelBitMask
from mangadap.proc.spectralfitting import EmissionLineFit
from mangadap.proc.spectralindices import SpectralIndices

def _spec_err(ivar, mask, calibrate = 1.0):
    mask_3d = np.array([mask] * ivar.shape[0])
    ivar[mask_3d == 0] = np.nan
    tot_var = np.nansum(1 / ivar, axis = (1, 2))
    N = np.sum(mask)
    return (1 + calibrate * np.log10(N)) * np.sqrt(tot_var)

def stack_spec_re(galaxy):
    flux, ivar = galaxy.get_cube(return_ivar=True)
    mask_re = utils.re_mask(galaxy).to_image(flux.shape[1:])
    spec = np.nansum(flux * mask_re, axis=(1, 2))
    if galaxy.weave_obs: spec_err = _spec_err(ivar, mask_re, calibrate=1.28)
    if galaxy.manga_obs: spec_err = _spec_err(ivar, mask_re, calibrate=1.62)
    wave = galaxy.get_spectrum(0, 0, only_wave=True)
    return wave, spec, spec_err

class FitSingleSpec:
    def __init__(self, galaxy):
        self.galaxy = galaxy
        wave, flux, flux_err = stack_spec_re(self.galaxy)
        self.wave = wave
        self.flux = flux.reshape(1, -1)
        self.flux_err = flux_err.reshape(1, -1)
        self.sres = np.repeat(2500, self.flux.size).reshape(1, -1)
        self.z = self.galaxy.z

    def write_output_file(self, wave, model_flux, line_model_cont, stellar_model_cont, line_flux, line_flux_err, line_ew, line_ew_err,
                           line_names, spec_index, spec_index_err, index_names):
        hdr = fits.Header()
        hdr["CLIFSID"] = self.galaxy.clifs_id
        hdu_primary = fits.PrimaryHDU(header = hdr)
        # Model
        col_wave = fits.Column(name="wave", format="D", array = wave)
        col_model_flux = fits.Column(name="model_flux", format="D", array = model_flux)
        col_line_cont = fits.Column(name="eline_cont", format="D", array = line_model_cont)
        col_stellar_cont = fits.Column(name="stellar_cont", format="D", array = stellar_model_cont)
        coldefs = fits.ColDefs([col_wave, col_model_flux, col_line_cont, col_stellar_cont])
        hdu_model = fits.BinTableHDU.from_columns(coldefs, name="MODEL")
        # Emission Lines
        col_line_name = fits.Column(name="name", format="10A", array = line_names)
        col_line_flux = fits.Column(name="flux", format="D", array = line_flux)
        col_line_flux_err = fits.Column(name="flux_err", format="D", array = line_flux_err)
        col_line_ew = fits.Column(name="ew", format="D", array = line_ew)
        col_line_ew_err = fits.Column(name="ew_err", format="D", array = line_ew_err)
        coldefs = fits.ColDefs([col_line_name, col_line_flux, col_line_flux_err, col_line_ew, col_line_ew_err])
        hdu_line = fits.BinTableHDU.from_columns(coldefs, name = "EMLINE")
        # Spectral Indices
        col_spec_name = fits.Column(name="name", format="10A", array = index_names)
        col_spec_ind = fits.Column(name="value", format="D", array = spec_index)
        col_spec_ind_err = fits.Column(name="error", format="D", array = spec_index_err)
        coldefs = fits.ColDefs([col_spec_name, col_spec_ind, col_spec_ind_err])
        hdu_spec = fits.BinTableHDU.from_columns(coldefs, name="SPECINDEX")
        # Write out
        outdir = "/arc/projects/CLIFS/dap_output/clifs/stack_re"
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
        hdul = fits.HDUList([hdu_primary, hdu_model, hdu_line, hdu_spec])
        hdul.writeto(f"{outdir}/clifs{self.galaxy.clifs_id}.fits", overwrite=True)

    def resample_to_log(self):
        wlum = self.wave.to(units.um).value
        wave_vac = ((1+1e-6*(287.6155+1.62887/wlum**2+0.01360/wlum**4)) * self.wave).to(units.AA).value
        dlogl = np.mean(np.diff(np.log10(wave_vac)))
        r = Resample(self.flux, e=self.flux_err, x=wave_vac, inLog=False, newRange=wave_vac[[0,-1]], newLog=True, newdx=dlogl)
        return r.outx, r.outy, r.oute

    def run_fit(self, usr_plots=False, fit_plots=False):
        if self.galaxy.weave_obs:
            wave, flux, ferr = self.resample_to_log()
        else:
            wave = self.wave.value
            flux = self.flux
            ferr = self.flux_err
        ivar = np.power(ferr, -2)
        spectral_step = spectral_coordinate_step(wave, log=True)
        drpall_file = "/arc/projects/CLIFS/drpall-v3_1_1.fits"
        dispersion = np.array([100.])
        # Templates used in the stellar continuum fits
        sc_tpl_key = 'MILESHC'
        # Templates used in the emission-line modeling
        el_tpl_key = 'MASTARSSP'
        # Template pixel scale a factor of 4 smaller than galaxy data
        sc_velscale_ratio = 4
        # Template sampling is the same as the galaxy data
        el_velscale_ratio = 1
        elmom_key = 'ELBMPL9'
        elfit_key = 'ELPMPL11'
        absindx_key = 'EXTINDX'
        bhdindx_key = 'BHBASIC'
        sc_pixel_mask = SpectralPixelMask(artdb=ArtifactDB.from_key('BADSKY'),
                                      emldb=EmissionLineDB.from_key('ELPMPL11'))
        # Mask the 5577 sky line
        el_pixel_mask = SpectralPixelMask(artdb=ArtifactDB.from_key('BADSKY'))
        # Fit the stellar continuum
        sc_tpl = TemplateLibrary(sc_tpl_key, match_resolution=False, velscale_ratio=sc_velscale_ratio,
                             spectral_step=spectral_step, log=True, hardcopy=False)
        sc_tpl_sres = np.mean(sc_tpl['SPECRES'].data, axis=0).ravel()
        ppxf = PPXFFit(StellarContinuumModelBitMask())
        cont_wave, cont_flux, cont_mask, cont_par = ppxf.fit(sc_tpl['WAVE'].data.copy(), sc_tpl['FLUX'].data.copy(), wave, flux, ferr,
                                                           self.z, dispersion, iteration_mode='no_global_wrej', reject_boxcar=100,
                                                           ensemble=False, velscale_ratio=sc_velscale_ratio, mask=sc_pixel_mask,
                                                           matched_resolution=False, tpl_sres=sc_tpl_sres, obj_sres=self.sres, degree=8,
                                                           moments=2, plot=fit_plots)
        # Remask the continuum fit
        sc_continuum = StellarContinuumModel.reset_continuum_mask_window(ma.MaskedArray(cont_flux, mask=cont_mask>0))
        # Show the fit and residual
        if usr_plots:
            pyplot.plot(wave, flux[0,:], label='Data')
            pyplot.plot(wave, sc_continuum[0,:], label='Model')
            pyplot.plot(wave, flux[0,:] - sc_continuum[0,:], label='Resid')
            pyplot.legend()
            pyplot.xlabel('Wavelength')
            pyplot.ylabel('Flux')
            pyplot.savefig('cont_fit.pdf')
        # Get the emission-line moments using the fitted stellar continuum
        momdb = EmissionMomentsDB.from_key(elmom_key)
        # Measure the moments
        elmom = EmissionLineMoments.measure_moments(momdb, wave, flux, continuum=sc_continuum, redshift=[self.z])
        # Fit the emission-line model
        # Set the emission-line continuum templates if different from those
        # used for the stellar continuum
        if sc_tpl_key == el_tpl_key:
            el_tpl = sc_tpl
            el_tpl_sres = sc_tpl_sres
            stellar_kinematics = cont_par['KIN']
        else:
            # If the template sets are different, we need to match the
            # spectral resolution to the galaxy data ...
            _sres = SpectralResolution(wave, self.sres[0,:], log10=True)
            el_tpl = TemplateLibrary(el_tpl_key, sres=_sres, velscale_ratio=el_velscale_ratio,
                                     spectral_step=spectral_step, log=True, hardcopy=False)
            el_tpl_sres = np.mean(el_tpl['SPECRES'].data, axis=0).ravel()
            stellar_kinematics = cont_par['KIN']
            stellar_kinematics[:,1] = ma.sqrt(np.square(cont_par['KIN'][:,1]) -
                                                        np.square(cont_par['SIGMACORR_EMP']))
        # Read the emission line fitting database
        emldb = EmissionLineDB.from_key(elfit_key)
        # Instantiate the fitting class
        emlfit = Sasuke(EmissionLineModelBitMask())
        # Perform the fit
        efit_t = time.perf_counter()
        eml_wave, model_flux, eml_flux, eml_mask, eml_fit_par, eml_eml_par \
                = emlfit.fit(emldb, wave, flux, obj_ferr=ferr, obj_mask=el_pixel_mask, obj_sres=self.sres,
                             guess_redshift=self.z, guess_dispersion=dispersion, reject_boxcar=101,
                             stpl_wave=el_tpl['WAVE'].data, stpl_flux=el_tpl['FLUX'].data,
                             stpl_sres=el_tpl_sres, stellar_kinematics=stellar_kinematics,
                             etpl_sinst_mode='offset', etpl_sinst_min=10.,
                             velscale_ratio=el_velscale_ratio, #matched_resolution=False,
                             mdegree=8,
                             plot=fit_plots)
        print('TIME: ', time.perf_counter() - efit_t)
        # Line-fit metrics
        eml_eml_par = EmissionLineFit.line_metrics(emldb, wave, flux, ferr, model_flux, eml_eml_par,
                                                   model_mask=eml_mask, bitmask=emlfit.bitmask)
        # Equivalent width
        EmissionLineFit.measure_equivalent_width(wave, flux, emldb, eml_eml_par, bitmask=emlfit.bitmask)
        # Get the stellar continuum that was fit for the emission lines
        elcmask = eml_mask.ravel() > 0
        goodpix = np.arange(elcmask.size)[np.invert(elcmask)]
        start, end = goodpix[0], goodpix[-1]+1
        elcmask[start:end] = False
        el_continuum = ma.MaskedArray(model_flux - eml_flux,
                                            mask=elcmask.reshape(model_flux.shape))
        # Plot the result
        if usr_plots:
            pyplot.plot(wave, flux[0,:], label='Data')
            pyplot.plot(wave, model_flux[0,:], label='Model')
            pyplot.plot(wave, el_continuum[0,:], label='EL Cont.')
            pyplot.plot(wave, sc_continuum[0,:], label='SC Cont.')
            pyplot.legend()
            pyplot.xlabel('Wavelength')
            pyplot.ylabel('Flux')
            pyplot.savefig('el_cont_fit.pdf')
        # Remeasure the emission-line moments with the new continuum
        new_elmom = EmissionLineMoments.measure_moments(momdb, wave, flux, continuum=el_continuum,
                                                        redshift=[self.z])
        # Compare the summed flux and Gaussian-fitted flux for all the
        # fitted lines
        if usr_plots:
            pyplot.scatter(emldb['restwave'], (new_elmom['FLUX']-eml_eml_par['FLUX']).ravel(),
                           c=eml_eml_par['FLUX'].ravel(), cmap='viridis', marker='.', s=60, lw=0,
                           zorder=4)
            pyplot.grid()
            pyplot.xlabel('Wavelength')
            pyplot.ylabel('Summed-Gaussian Difference')
            pyplot.savefig('flux_comparison.pdf')
        # Measure the spectral indices
        if absindx_key is None or bhdindx_key is None:
            # Neither are defined, so we're done
            print('Elapsed time: {0} seconds'.format(time.perf_counter() - t))
            return
        # Setup the databases that define the indices to measure
        absdb = None if absindx_key is None else AbsorptionIndexDB.from_key(absindx_key)
        bhddb = None if bhdindx_key is None else BandheadIndexDB.from_key(bhdindx_key)
        # Remove the modeled emission lines from the spectra
        flux_noeml = flux - eml_flux
        redshift = stellar_kinematics[:,0] / astropy.constants.c.to('km/s').value
        sp_indices = SpectralIndices.measure_indices(absdb, bhddb, wave, flux_noeml, ivar=ivar,
                                                     redshift=redshift)
        # Calculate the velocity dispersion corrections
        #   - Construct versions of the best-fitting model spectra with and without
        #     the included dispersion
        continuum = Sasuke.construct_continuum_models(emldb, el_tpl['WAVE'].data, el_tpl['FLUX'].data,
                                                      wave, flux.shape, eml_fit_par)
        continuum_dcnvlv = Sasuke.construct_continuum_models(emldb, el_tpl['WAVE'].data,
                                                             el_tpl['FLUX'].data, wave, flux.shape,
                                                             eml_fit_par, redshift_only=True)
        #   - Get the dispersion corrections and fill the relevant columns of the
        #     index table
        sp_indices['BCONT_MOD'], sp_indices['BCONT_CORR'], sp_indices['RCONT_MOD'], \
            sp_indices['RCONT_CORR'], sp_indices['MCONT_MOD'], sp_indices['MCONT_CORR'], \
            sp_indices['AWGT_MOD'], sp_indices['AWGT_CORR'], \
            sp_indices['INDX_MOD'], sp_indices['INDX_CORR'], \
            sp_indices['INDX_BF_MOD'], sp_indices['INDX_BF_CORR'], \
            good_les, good_ang, good_mag, is_abs \
                    = SpectralIndices.calculate_dispersion_corrections(absdb, bhddb, wave, flux,
                                                                       continuum, continuum_dcnvlv,
                                                                       redshift=redshift,
                                                                       redshift_dcnvlv=redshift)
        # Apply the index corrections.  This is only done here for the
        # Worthey/Trager definition of the indices, as an example
        corrected_indices = np.zeros(sp_indices['INDX'].shape, dtype=float)
        corrected_indices_err = np.zeros(sp_indices['INDX'].shape, dtype=float)
        # Unitless indices
        corrected_indices[good_les], corrected_indices_err[good_les] \
                = SpectralIndices.apply_dispersion_corrections(sp_indices['INDX'][good_les],
                                                               sp_indices['INDX_CORR'][good_les],
                                                               err=sp_indices['INDX_ERR'][good_les])
        # Indices in angstroms
        corrected_indices[good_ang], corrected_indices_err[good_ang] \
                = SpectralIndices.apply_dispersion_corrections(sp_indices['INDX'][good_ang],
                                                               sp_indices['INDX_CORR'][good_ang],
                                                               err=sp_indices['INDX_ERR'][good_ang],
                                                               unit='ang')
        # Indices in magnitudes
        corrected_indices[good_mag], corrected_indices_err[good_mag] \
                = SpectralIndices.apply_dispersion_corrections(sp_indices['INDX'][good_mag],
                                                               sp_indices['INDX_CORR'][good_mag],
                                                               err=sp_indices['INDX_ERR'][good_mag],
                                                               unit='mag')
        # Write results to file
        self.write_output_file(wave, model_flux[0], el_continuum[0], sc_continuum[0], eml_eml_par["FLUX"][0], eml_eml_par["FLUXERR"][0],
                           eml_eml_par["EW"][0], eml_eml_par["EWERR"][0], emldb["name"], corrected_indices[0], corrected_indices_err[0], absdb["name"])

if __name__ == "__main__":
    from clifspy.galaxy import Galaxy
    import matplotlib.pyplot as plt

    gal = Galaxy(34)
    fit = FitSingleSpec(gal)
    fit.run_fit()
