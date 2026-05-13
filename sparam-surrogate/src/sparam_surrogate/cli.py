#! /usr/bin/env python3
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
import sys
from pathlib import Path
from typing import Sequence

import sparam_surrogate.config.mylogging as mylogging
from sparam_surrogate import utils, __app_name__, __version__


# %%
class CLI:
    """
    Command-line interface for sparam-surrogate.
    """

    def __init__(self, prog: str|None = None, desc: str|None = None):
        self._app_name = prog if prog else __app_name__
        self._desc = desc if desc else "A ML surrogate for S-parameter prediction."

        # Initialize the argument parser
        self._parser = argparse.ArgumentParser(
            prog=self._app_name,
            description=self._desc
        )
        self._subparsers = self._parser.add_subparsers(dest="command")

        # Add version argument
        self._parser.add_argument(
            "-v", "--version",
            action = "version",
            version= f"{self._app_name} v{__version__}"
        )

    @property
    def app_name(self) -> str:
        """
        Get the application name.
        """
        return self._app_name

    def add_subcommand_unzip(self) -> None:
        """
        Add the unzip subcommand to the CLI parser.
        """

        unzip_parser = self._subparsers.add_parser(
            "unzip",
            help="Extract a ZIP archive.",
            description="Extract a ZIP archive to an output directory.",
        )
        unzip_parser.add_argument(
            "-i", "--infile",
            required=True,
            type=Path,
            metavar="<zip file>",
            help="Path to the ZIP archive to extract."
        )
        unzip_parser.add_argument(
            "-o", "--outdir",
            required=True,
            type=Path,
            metavar="<destination directory>",
            help="Directory where files should be extracted to."
        )

    def add_subcommand_preprocess(self) -> None:
        """
        Add the preprocess subcommand to the CLI parser.
        """
        preproc_parser = self._subparsers.add_parser(
            "preprocess",
            help="Preprocess raw S-parameter data.",
            description="Preprocess raw S-parameter data for surrogate modeling."
        )
        preproc_parser.add_argument(
            "-m", "--model",
            choices=["decision_tree", "scalar_nn", "vector_nn", "smatrix_nn"],
            required=True,
            type=str,
            metavar="<model name>",
            help="The target model to which the preprocessed data will be fed."
        )
        preproc_parser.add_argument(
            "-i", "--input-dir",
            required=True,
            type=Path,
            metavar="<input directory>",
            help="Directory containing raw S-parameter data files."
        )
        preproc_parser.add_argument(
            "-o", "--output-dir",
            required=True,
            type=Path,
            metavar="<output directory>",
            help="Directory where preprocessed data should be saved."
        )

    def add_subcommand_train(self) -> None:
        """
        Add the train subcommand to the CLI parser.
        """
        train_parser = self._subparsers.add_parser(
            "train",
            help="Train a surrogate model.",
            description="Train a surrogate model using preprocessed S-parameter data."
        )
        train_parser.add_argument(
            "-m", "--model",
            choices=["decision_tree", "scalar_nn", "vector_nn", "smatrix_nn"],
            required=True,
            type=str,
            metavar="<model name>",
            help="The surrogate model architecture to train."
        )
        train_parser.add_argument(
            "-i", "--input-dir",
            required=True,
            type=Path,
            metavar="<input directory>",
            help="Directory containing preprocessed data files."
        )
        train_parser.add_argument(
            "-o", "--output-dir",
            required=True,
            type=Path,
            metavar="<output directory>",
            help="Directory where trained model and logs should be saved."
        )

    def add_subcommand_predict(self) -> None:
        """
        Add the predict subcommand to the CLI parser.
        """
        pred_parser = self._subparsers.add_parser(
            "predict",
            help="Make predictions with a trained surrogate model.",
            description="Use a trained surrogate model to make predictions."
        )
        pred_parser.add_argument(
            "-m", "--model",
            choices=["decision_tree", "scalar_nn", "vector_nn", "smatrix_nn"],
            required=True,
            type=str,
            metavar="<model name>",
            help="The surrogate model architecture to use for prediction."
        )
        pred_parser.add_argument(
            "-f", "--from-file",
            action="store_true",
            help="Whether input json cames from a file or CLI arguments."
        )
        pred_parser.add_argument(
            "input",
            type=str,
            metavar="<input data, JSON string or file name>",
            help=("Input data for prediction, either as a JSON string or a path"
                  " to a JSON file (if --from-file is set)."
            )
        )

    def parse_cli(
        self, args: Sequence[str] | None = None, namespace: None = None
    ) -> argparse.Namespace:
        """
        Parse the command-line arguments and return the parsed arguments.
        """
        return self._parser.parse_args(args, namespace)


# %%
def main()-> int:
    """
    Main function for the CLI.
    """

    # Set up logging
    mylogging.set_logging_cfg()
    logger = mylogging.get_md_logger("sparam_surrogate.cli")

    logger.debug("Starting %s CLI, version %s", __app_name__, __version__)
    logger.debug("Initial CLI arguments: %s", " ".join(sys.argv))

    # Parse CLI inputs
    parser = CLI(prog = f"{__app_name__}")
    parser.add_subcommand_unzip()
    parser.add_subcommand_preprocess()
    parser.add_subcommand_train()
    parser.add_subcommand_predict()
    cli = parser.parse_cli()

    try:
        match cli.command:
            case "unzip":
                dest = utils.extract_zip(cli.infile, cli.outdir)
                logger.info("Extracted %s to %s", cli.infile, dest)
                return 0
            case "preprocess":
                # TODO
                logger.info("Preprocessed data saved to %s", cli.output_dir)
                return 0
            case "train":
                # TODO
                logger.info("Trained model saved to %s", cli.output_dir)
                return 0
            case "predict":
                # TODO
                logger.info("Prediction made for input: %s", cli.input)
                return 0
    except Exception as e:
        logger.error("Error executing command '%s': %s", cli.command, str(e))
        return 1

    print(f"{__app_name__} version {__version__}")
    print("A ML surrogate for S-parameter prediction.")
    return 0
