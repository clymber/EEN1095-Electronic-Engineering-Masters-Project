#!/usr/bin/env python3
"""
Behavior-focused tests for the sparam-surrogate CLI.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from sparam_surrogate import __app_name__, __version__
from sparam_surrogate.cli import CLI, main


class FakePointwiseDataset:
    """
    Test double for ``PointwiseDataset`` CSV construction.
    """

    build_calls: list[dict[str, object]] = []

    @classmethod
    def build_frequency_expanded_csv(
        cls,
        split_parameter_csv: Path,
        output_csv: Path,
        force: bool = False,
    ) -> None:
        """
        Record frequency-expanded CSV construction.
        """
        cls.build_calls.append(
            {
                "split_parameter_csv": split_parameter_csv,
                "output_csv": output_csv,
                "force": force,
            }
        )


class FakeParameterBuilder:
    """
    Test double for ``ParameterDatasetBuilder``.
    """

    instances: list["FakeParameterBuilder"] = []

    def __init__(self, raw_data: object, cleaned_splits_path: Path) -> None:
        """
        Record design-level builder construction.
        """
        self.raw_data = raw_data
        self.cleaned_splits_path = cleaned_splits_path
        self.build_kwargs: dict[str, object] | None = None
        self.instances.append(self)

    def build(
        self,
        *,
        val_fraction: float,
        test_fraction: float,
        seed: int,
        force: bool,
    ) -> object:
        """
        Record cache-aware build arguments and return a placeholder dataframe.
        """
        self.build_kwargs = {
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
            "seed": seed,
            "force": force,
        }
        return object()


class FakeRawData:
    """
    Test double for ``RawData``.
    """

    instances: list["FakeRawData"] = []

    def __init__(self, path: Path, nports: int) -> None:
        """
        Record construction arguments for assertions.
        """
        self.path = path
        self.nports = nports
        self.instances.append(self)


class TestCLI:
    """
    Unit tests for CLI behavior.
    """

    def test_version_argument(self, capsys: pytest.CaptureFixture[str]) -> None:
        """
        Version argument reports the application version.
        """
        with pytest.raises(SystemExit) as exc_info:
            CLI().parse_cli(["--version"])
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert captured.out == f"{__app_name__} v{__version__}\n"

    def test_preprocess_defaults_come_from_typed_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """
        Preprocess defaults are read from ``SurrogateConfig``.
        """
        cfg = SimpleNamespace(
            dataset=SimpleNamespace(nports=12),
            preprocessing=SimpleNamespace(
                cleaned_splits_csv=Path(
                    "data/processed/cleaned_splits_parameter.csv"
                ),
                freq_expanded_csv=Path(
                    "data/processed/frequency_expanded_dataset.csv"
                ),
                val_fraction=0.25,
                test_fraction=0.1,
            ),
            project=SimpleNamespace(seed=99),
        )
        self._patch_preprocess_dependencies(monkeypatch, cfg)
        monkeypatch.setattr(
            "sys.argv",
            [
                "sparam-surrogate",
                "preprocess",
                "-i",
                str(tmp_path / "raw"),
                "-o",
                str(tmp_path / "processed"),
            ],
        )

        exit_code = main()

        assert exit_code == 0
        assert FakeRawData.instances[0].nports == 12
        assert FakeParameterBuilder.instances[0].build_kwargs == {
            "val_fraction": 0.25,
            "test_fraction": 0.1,
            "seed": 99,
            "force": False,
        }
        assert FakePointwiseDataset.build_calls == [
            {
                "split_parameter_csv": (
                    tmp_path / "processed" / "cleaned_splits_parameter.csv"
                ),
                "output_csv": (
                    tmp_path / "processed" / "frequency_expanded_dataset.csv"
                ),
                "force": False,
            }
        ]

    def test_preprocess_cli_values_override_typed_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """
        Explicit preprocess CLI options override typed config defaults.
        """
        cfg = SimpleNamespace(
            dataset=SimpleNamespace(nports=12),
            preprocessing=SimpleNamespace(
                cleaned_splits_csv=Path(
                    "data/processed/cleaned_splits_parameter.csv"
                ),
                freq_expanded_csv=Path(
                    "data/processed/frequency_expanded_dataset.csv"
                ),
                val_fraction=0.25,
                test_fraction=0.1,
            ),
            project=SimpleNamespace(seed=99),
        )
        self._patch_preprocess_dependencies(monkeypatch, cfg)
        monkeypatch.setattr(
            "sys.argv",
            [
                "sparam-surrogate",
                "preprocess",
                "-i",
                str(tmp_path / "raw"),
                "-o",
                str(tmp_path / "processed"),
                "--nports",
                "6",
                "--val-fraction",
                "0.2",
                "--test-fraction",
                "0.15",
                "--seed",
                "123",
                "--force",
            ],
        )

        exit_code = main()

        assert exit_code == 0
        assert FakeRawData.instances[0].nports == 6
        assert FakeParameterBuilder.instances[0].build_kwargs == {
            "val_fraction": 0.2,
            "test_fraction": 0.15,
            "seed": 123,
            "force": True,
        }
        assert FakePointwiseDataset.build_calls == [
            {
                "split_parameter_csv": (
                    tmp_path / "processed" / "cleaned_splits_parameter.csv"
                ),
                "output_csv": (
                    tmp_path / "processed" / "frequency_expanded_dataset.csv"
                ),
                "force": True,
            }
        ]

    def _patch_preprocess_dependencies(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cfg: SimpleNamespace,
    ) -> None:
        """
        Replace preprocessing dependencies with deterministic test doubles.
        """
        FakePointwiseDataset.build_calls = []
        FakeParameterBuilder.instances = []
        FakeRawData.instances = []
        monkeypatch.setattr(
            "sparam_surrogate.cli.SurrogateConfig.from_config",
            lambda: cfg,
        )
        monkeypatch.setattr("sparam_surrogate.cli.RawData", FakeRawData)
        monkeypatch.setattr(
            "sparam_surrogate.cli.PointwiseDataset",
            FakePointwiseDataset,
        )
        monkeypatch.setattr(
            "sparam_surrogate.cli.ParameterDatasetBuilder",
            FakeParameterBuilder,
        )
