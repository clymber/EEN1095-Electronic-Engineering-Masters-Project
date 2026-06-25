#! /usr/bin/env python3
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
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
from collections.abc import Sequence
from pathlib import Path

import sparam_surrogate.config.mylogging as mylogging
from sparam_surrogate import __app_name__, __version__, utils
from sparam_surrogate.config import SurrogateConfig
from sparam_surrogate.data import MLDatasetBuilder, RawData


# %%
class CLI:
    """
    Command-line interface for sparam-surrogate.
    """

    def __init__(self, prog: str | None = None, desc: str | None = None):
        self._app_name = prog if prog else __app_name__
        self._desc = desc if desc else "A ML surrogate for S-parameter prediction."

        # Initialize the argument parser
        self._parser = argparse.ArgumentParser(
            prog=self._app_name, description=self._desc
        )
        self._subparsers = self._parser.add_subparsers(dest="command")

        # Add version argument
        self._parser.add_argument(
            "-v",
            "--version",
            action="version",
            version=f"{self._app_name} v{__version__}",
        )

    @property
    def app_name(self) -> str:
        """
        Return the application name.
        """
        return self._app_name

    def add_subcommand_unzip(self) -> None:
        """
        Add the unzip subcommand.
        """
        unzip_parser = self._subparsers.add_parser(
            "unzip",
            help="Extract a ZIP archive.",
            description="Extract a ZIP archive to an output directory.",
        )
        unzip_parser.add_argument(
            "-i",
            "--infile",
            required=True,
            type=Path,
            metavar="<zip file>",
            help="Path to the ZIP archive to extract.",
        )
        unzip_parser.add_argument(
            "-o",
            "--outdir",
            required=True,
            type=Path,
            metavar="<destination directory>",
            help="Directory where files should be extracted to.",
        )

    def add_subcommand_preprocess(self) -> None:
        """
        Add the cleaned-CSV preprocessing subcommand to the CLI parser.

        The command builds ``sipi_dataset_cleaned.csv`` and assigns
        train/validation/test split labels. It does not create model-specific
        eager arrays.
        """
        preproc_parser = self._subparsers.add_parser(
            "preprocess",
            help="Build the cleaned lazy preprocessing CSV.",
            description=("Build sipi_dataset_cleaned.csv from one raw SI/PI topology."),
        )
        preproc_parser.add_argument(
            "-i",
            "--input-dir",
            required=True,
            type=Path,
            metavar="<input directory>",
            help="Directory containing raw S-parameter data files.",
        )
        preproc_parser.add_argument(
            "-o",
            "--output-dir",
            required=True,
            type=Path,
            metavar="<output directory>",
            help="Directory where preprocessed data should be saved.",
        )
        preproc_parser.add_argument(
            "--nports",
            type=int,
            default=None,
            metavar="<port count>",
            help="Expected Touchstone port count. Defaults to configs/default.json.",
        )
        preproc_parser.add_argument(
            "--val-fraction",
            type=float,
            default=None,
            metavar="<val fraction>",
            help="Validation split fraction. Defaults to configs/default.json.",
        )
        preproc_parser.add_argument(
            "--test-fraction",
            type=float,
            default=None,
            metavar="<test fraction>",
            help="Test split fraction. Defaults to configs/default.json.",
        )
        preproc_parser.add_argument(
            "--seed",
            type=int,
            default=None,
            metavar="<random seed>",
            help="Split random seed. Defaults to configs/default.json.",
        )

    def add_subcommand_train(self) -> None:
        """
        Add the train subcommand.
        """
        train_parser = self._subparsers.add_parser(
            "train",
            help="Train a surrogate model.",
            description="Train a surrogate model using preprocessed S-parameter data.",
        )
        train_parser.add_argument(
            "-m",
            "--model",
            choices=["decision_tree", "scalar_nn", "vector_nn", "smatrix_nn"],
            required=True,
            type=str,
            metavar="<model name>",
            help="The surrogate model architecture to train.",
        )
        train_parser.add_argument(
            "-i",
            "--input-dir",
            required=True,
            type=Path,
            metavar="<input directory>",
            help="Directory containing preprocessed data files.",
        )
        train_parser.add_argument(
            "-o",
            "--output-dir",
            required=True,
            type=Path,
            metavar="<output directory>",
            help="Directory where trained model and logs should be saved.",
        )

    def add_subcommand_predict(self) -> None:
        """
        Add the predict subcommand.
        """
        pred_parser = self._subparsers.add_parser(
            "predict",
            help="Make predictions with a trained surrogate model.",
            description="Use a trained surrogate model to make predictions.",
        )
        pred_parser.add_argument(
            "-m",
            "--model",
            choices=["decision_tree", "scalar_nn", "vector_nn", "smatrix_nn"],
            required=True,
            type=str,
            metavar="<model name>",
            help="The surrogate model architecture to use for prediction.",
        )
        pred_parser.add_argument(
            "-f",
            "--from-file",
            action="store_true",
            help="Whether input json cames from a file or CLI arguments.",
        )
        pred_parser.add_argument(
            "input",
            type=str,
            metavar="<input data, JSON string or file name>",
            help=(
                "Input data for prediction, either as a JSON string or a path"
                " to a JSON file (if --from-file is set)."
            ),
        )

    def parse_cli(
        self, args: Sequence[str] | None = None, namespace: None = None
    ) -> argparse.Namespace:
        """
        Parse command-line arguments.
        """
        return self._parser.parse_args(args, namespace)


# %%
def main() -> int:
    """
    Run the command-line interface.
    """
    # Set up logging
    mylogging.set_logging_cfg()
    logger = mylogging.get_md_logger("sparam_surrogate.cli")

    logger.debug("Starting %s CLI, version %s", __app_name__, __version__)
    logger.debug("Initial CLI arguments: %s", " ".join(sys.argv))

    # Parse CLI inputs
    parser = CLI(prog=f"{__app_name__}")
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
                cfg = SurrogateConfig.from_csv()
                nports = (
                    cli.nports
                    if cli.nports is not None
                    else cfg.dataset.nports
                )
                val_fraction = (
                    cli.val_fraction
                    if cli.val_fraction is not None
                    else cfg.preprocessing.val_fraction
                )
                test_fraction = (
                    cli.test_fraction
                    if cli.test_fraction is not None
                    else cfg.preprocessing.test_fraction
                )
                seed = cli.seed if cli.seed is not None else cfg.project.seed
                raw_data = RawData(cli.input_dir, nports=nports)
                builder = MLDatasetBuilder(raw_data, cli.output_dir)
                builder.split(
                    val_fraction=val_fraction,
                    test_fraction=test_fraction,
                    seed=seed,
                )
                logger.info("Preprocessed CSV saved to %s", builder.cleaned_path)
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
