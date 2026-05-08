#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: meng
#     language: python
#     name: python3
# ---

# %%
"""
CLI for sparam-surrogate
"""
import argparse, sys
from sparam_surrogate import __version__
import sparam_surrogate.mylogging as mylogging


# %%
def main()-> None:
    """
    Main function for the CLI.
    """

    mylogging.set_logging_cfg()
    logger = mylogging.get_md_logger("sparam_surrogate.cli")

    logger.debug("Starting sparam-surrogate CLI, version %s", __version__)
    logger.debug("Initial CLI arguments: %s", " ".join(sys.argv))

    parser = argparse.ArgumentParser(
        prog = "sparam-surrogate",
        description = "A ML surrogate for S-parameter prediction."
    )

    parser.add_argument(
        "-v", "--version",
        action = "version",
        version= f"sparam-surrogate v{__version__}"
    )

    parser.parse_args()

    print("sparam-surrogate")
    print("A ML surrogate for S-parameter prediction.")
