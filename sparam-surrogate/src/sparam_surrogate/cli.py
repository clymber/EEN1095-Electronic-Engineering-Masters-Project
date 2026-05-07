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
import argparse
from sparam_surrogate import __version__

def main()-> None:
    """
    Main function for the CLI.
    """

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
