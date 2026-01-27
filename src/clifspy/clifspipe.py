import warnings
import os.path
import logging
import time
import argparse

from clifspy import (derived_products, cube, galaxy, dap, utils,
                        multiwav, plotting)

def setup_logger(gal_id):
    timestr = time.strftime("%Y%m%d-%H%M%S")

    logger = logging.getLogger("CLIFS_Pipeline")
    logger.setLevel(logging.DEBUG)

    format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(format)

    logfile = "/arc/projects/CLIFS/log_files/clifspipe/clifs{}_{}.log".format(gal_id, timestr)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(format)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.addHandler(fh)

    return logger, warnings_logger, logfile

def run_clifs_pipeline(args, logger):
    if not os.path.isfile("/arc/projects/CLIFS/config_files/clifs_{}.toml".format(args.clifs_id)):
        logger.info("Galaxy config file does not exist, generating it now")
        utils.clifs_config_file(args.clifs_id).make()
    this_galaxy = galaxy.Galaxy(args.clifs_id)

    if args.process_cube:
        logger.info("STARTING PROCESS CUBE...")
        cube.generate_cube(this_galaxy)
        cube.generate_cube(this_galaxy, fullfield = True)
        logger.info("DONE PROCESS CUBE")

    if args.multiwav:
        logger.info("STARTING MULTIWAVELENGTH PRODUCTS")
        multiwav.make_multiwav_cutouts(this_galaxy)
        logger.info("DONE MULTIWAVELENGTH PRODUCTS")

    #if "RADSTACK" in steps:
    #    logger.info("STARTING RADSTACK...")
    #    stack_spectrum_radial(galaxy)
    #    logger.info("DONE RADSTACK")

    if args.manga_dap:
        logger.info("Starting MANGA DAP...")
        dap.run_manga_dap(this_galaxy, decompress = True)
        logger.info("Done MANGA DAP")

    if args.products:
        logger.info("STARTING DERIVED PRODUCTS")
        derived_products.products_for_clifspipe(this_galaxy)
        logger.info("DONE DERVIVED PRODUCTS")

    if args.make_plots:
        logger.info("Making plots...")
        plotting.plots_for_clifspipe(this_galaxy)
        logger.info("Done plotting")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clifs_id", type = int)
    parser.add_argument("--default_run", action = "store_true")
    parser.add_argument("--process_cube", action = "store_true")
    parser.add_argument("--multiwav", action = "store_true")
    parser.add_argument("--manga_dap", action = "store_true")
    parser.add_argument("--products", action = "store_true")
    parser.add_argument("--make_plots", action = "store_true")
    parser.add_argument("--xmin", type = int, default = -99)
    parser.add_argument("--xmax", type = int, default = -99)
    parser.add_argument("--ymin", type = int, default = -99)
    parser.add_argument("--ymax", type = int, default = -99)
    parser.add_argument("--bkgsub", default = "true")
    parser.add_argument("--bkgsub_galmask", default = "true")
    parser.add_argument("--downsample_spatial", default = "true")
    parser.add_argument("--alpha", type = float, default = 1.28)
    parser.add_argument("--factor_spatial", type = int, default = 2)
    parser.add_argument("--downsample_wav", default = "true")
    parser.add_argument("--fill_ccd_gaps", default = "false")
    parser.add_argument("--fix_astrometry", default = "false")
    parser.add_argument("--hdf5", default = "true")
    parser.add_argument("--verbose", default = "false")
    parser.add_argument("--clobber", default = "true")
    args = parser.parse_args()

    if args.default_run & (args.process_cube | args.multiwav | args.manga_dap | args.make_plots):
        msg = "If the 'default_run' flag is provided, then no other pipeline components can be selected"
        raise ValueError(msg)

    logger, warnings_logger, logfile = setup_logger(args.clifs_id)
    logger.info("CLIFS PROCESSING PIPELINE")
    if args.default_run:
        run_components = ["process_cube", "multiwav", "manga_dap", "make_plots"]
        args.process_cube = True
        args.multiwav = True
        args.manga_dap = True
        args.make_plots = True
        logger.info("Default pipeline run selected. Running the following components: {}".format(run_components))
    logger.info("Full log, including Python warnings, will be stored in: {}".format(logfile))
    run_clifs_pipeline(args, logger)
    logger.info("CLIFS PIPELINE HAS FINISHED")
