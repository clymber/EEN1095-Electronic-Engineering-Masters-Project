#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for class CLI in sparam_surrogate.cli.
"""

import pytest
from pathlib import Path
from sparam_surrogate.cli import CLI
from sparam_surrogate import __app_name__, __version__

class TestCLI:
    """
    Unit tests for class CLI.
    """

    def test_instantiation(self):
        """
        Test that the CLI class can be instantiated without errors.
        """
        # Default application name
        cli = CLI()
        assert cli.app_name == __app_name__

        # Custom application name
        cli2 = CLI(prog="testprog")
        assert cli2.app_name == "testprog"

    def test_version_argument(self, capsys):
        """
        Version argument is correctly set up in the CLI parser.
        """
        with pytest.raises(SystemExit) as exc_info:
            CLI().parse_cli(["--version"]) # or ["-v"]
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert captured.out == f"{__app_name__} v{__version__}\n"

    def test_add_subcmd_unzip(self):
        """
        Add subcommand: unzip

        Usage: sparam-surrogate -i|--infile <zip archive> -o|--outdir <out directory>
        """
        cli = CLI()
        cli.add_subcommand_unzip()
        args = cli.parse_cli(["unzip", "-i", "test.zip", "-o", "output/"])

        assert args.command == "unzip"
        assert args.infile == Path("test.zip")
        assert args.outdir == Path("output/")

    def test_add_subcommand_preprocess(self):
        """
        Test adding preprocess subcommand.

        Usage:
            sparam-surrogate preprocess -m|--model <model type> \
                                        -i|--input <input dir> \
                                        -o|--output <output dir>
        """
        cli = CLI()
        cli.add_subcommand_preprocess()
        args = cli.parse_cli([
            "preprocess",
            "-m", "scalar_nn",
            "-i", "input/",
            "-o", "output/"
        ])
        assert args.command == "preprocess"
        assert args.model == "scalar_nn"
        assert args.input_dir == Path("input/")
        assert args.output_dir == Path("output/")

    def test_add_subcommand_preprocess_invalid_model(self):
        """
        Test preprocess with invalid model choice.
        """
        cli = CLI()
        cli.add_subcommand_preprocess()
        with pytest.raises(SystemExit):
            cli.parse_cli([
                "preprocess",
                "-m", "invalid_model",
                "-i", "input/",
                "-o", "output/"
            ])

    def test_add_subcommand_train(self):
        """
        Test adding train subcommand.
        
        Usage: sparam-surrogate train -m|--model <model name> \
                                      -i|--input-dir <input directory> \
                                      -o|--output-dir <output directory>
        """
        cli = CLI()
        cli.add_subcommand_train()
        args = cli.parse_cli([
            "train",
            "-m", "vector_nn",
            "-i", "input/",
            "-o", "output/"
        ])
        assert args.command == "train"
        assert args.model == "vector_nn"
        assert args.input_dir == Path("input/")
        assert args.output_dir == Path("output/")

    def test_add_subcommand_predict(self):
        """
        Test adding predict subcommand.

        Usage:
            sparam-surrogate predict -m|--model <model name> <input json string>
        """
        cli = CLI()
        cli.add_subcommand_predict()
        args = cli.parse_cli([
            "predict",
            "-m", "smatrix_nn",
            '{"param": "value"}'
        ])
        assert args.command == "predict"
        assert args.model == "smatrix_nn"
        assert args.input == '{"param": "value"}'
        assert args.from_file is False

    def test_add_subcommand_predict_from_file(self):
        """
        Test predict subcommand with --from-file flag.
        
        Usage:
            sparam-surrogate predict -m|--model <model name> \
                                     -f|--from-file <input json file>
        """
        cli = CLI()
        cli.add_subcommand_predict()
        args = cli.parse_cli([
            "predict",
            "-m", "decision_tree",
            "--from-file",
            "/path/to/file.json"
        ])
        assert args.command == "predict"
        assert args.from_file is True
        assert args.input == "/path/to/file.json"
