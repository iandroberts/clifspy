import warnings
import os.path
import logging
import time
import argparse
from pathlib import Path
import toml

from clifspy import (derived_products, cube, galaxy, dap, utils,
                        multiwav, plotting)

def setup_logger(args):
    timestr = time.strftime("%Y%m%d-%H%M%S")

    logger = logging.getLogger("CLIFS_Pipeline")
    logger.setLevel(logging.DEBUG)

    format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(format)

    Path(f"{args.wkdir}/logs").mkdir(exist_ok=True)
    logfile = f"{args.wkdir}/logs/{timestr}.log"
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
    cfg_path = f"{args.wkdir}/{args.config}"
    cfg = toml.load(cfg_path)

    if args.process_cube:
        logger.info("STARTING PROCESS CUBE...")
        cube.generate_cube(cfg)
        logger.info("DONE PROCESS CUBE")

    if args.manga_dap:
        logger.info("Starting MANGA DAP...")
        dap.run_manga_dap(cfg, cfg_path, decompress=True)
        logger.info("Done MANGA DAP")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wkdir")
    parser.add_argument("config")
    parser.add_argument("config_dap")
    parser.add_argument("--process_cube", action = "store_true")
    parser.add_argument("--manga_dap", action = "store_true")
    args = parser.parse_args()

    logger, warnings_logger, logfile = setup_logger(args)
    logger.info("CLIFS PROCESSING PIPELINE")
    logger.info("Full log, including Python warnings, will be stored in: {}".format(logfile))
    run_clifs_pipeline(args, logger)
    logger.info("CLIFS PIPELINE HAS FINISHED")
