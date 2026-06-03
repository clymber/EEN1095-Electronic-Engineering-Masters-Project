"""
Orchestration for building model-ready scalar and full-S-matrix datasets.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .design_frequency_splitter import DesignFrequencySplitter
from .ml_dataset import MLDataset
from .pcb_feature_transformer import PcbFeatureMatrix, PcbFeatureTransformer
from .pcb_parameters import PcbParameters
from .s_parameter_dataset import SParameterDataset
from .target_builder import TargetBuilder, TargetMatrix


class MLDatasetBuilder:
    """
    Build ML-ready datasets from aligned parameters and S-parameter responses.
    """

    SCALAR_FILENAME = "scalar_baseline_dataset.npz"
    FULL_SMATRIX_FILENAME = "full_smatrix_dataset.npz"

    def __init__(
        self,
        splitter: DesignFrequencySplitter | None = None,
        feature_transformer: PcbFeatureTransformer | None = None,
        output_dir: Path | str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Configure dataset assembly.

        Parameters
        ----------
        splitter:
            Design-level train/validation/test splitter. When omitted, the
            default deterministic splitter is used.
        feature_transformer:
            Transformer that builds and scales
            ``X = [geometric/material parameters, frequency]``.
        output_dir:
            Optional directory where processed `.npz` datasets are saved.
        metadata:
            Optional JSON-serializable metadata included in every dataset.
        """
        self.splitter = splitter or DesignFrequencySplitter()
        self.feature_transformer = feature_transformer or PcbFeatureTransformer()
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.metadata = dict(metadata or {})

    def build_scalar_dataset(
        self,
        parameters: PcbParameters | pd.DataFrame,
        responses: SParameterDataset,
        pair: tuple[int, int],
        representation: str = "db",
    ) -> MLDataset:
        """
        Build the scalar baseline ML dataset.
        """
        features = self._build_features(parameters, responses)
        target = TargetBuilder.build_scalar(
            responses,
            pair=pair,
            representation=representation,
        )
        dataset = self._dataset(
            features,
            target,
            responses,
            {
                "target_mode": "scalar",
                "scalar_pair": [int(pair[0]), int(pair[1])],
                "scalar_representation": representation,
            },
        )
        self._save_if_requested(dataset, self.SCALAR_FILENAME)
        return dataset

    def build_full_smatrix_dataset(
        self,
        parameters: PcbParameters | pd.DataFrame,
        responses: SParameterDataset,
    ) -> MLDataset:
        """
        Build the full complex S-matrix ML dataset.
        """
        features = self._build_features(parameters, responses)
        target = TargetBuilder.build_full_smatrix(responses)
        dataset = self._dataset(
            features,
            target,
            responses,
            {"target_mode": "full_smatrix"},
        )
        self._save_if_requested(dataset, self.FULL_SMATRIX_FILENAME)
        return dataset

    def _build_features(
        self,
        parameters: PcbParameters | pd.DataFrame,
        responses: SParameterDataset,
    ) -> PcbFeatureMatrix:
        aligned_parameters = self._align_parameters(parameters, responses)
        split = self.splitter.split(responses.simulation_indices)
        design_labels = split.labels_for(responses.simulation_indices)
        return self.feature_transformer.fit_transform(
            aligned_parameters,
            responses.frequencies_ghz,
            design_labels,
        )

    def _dataset(
        self,
        features: PcbFeatureMatrix,
        target: TargetMatrix,
        responses: SParameterDataset,
        target_metadata: Mapping[str, Any],
    ) -> MLDataset:
        self._validate_row_alignment(features, target)
        split = self.splitter.split(responses.simulation_indices)
        split_labels = split.expand_labels(
            responses.simulation_indices,
            len(responses.frequencies_ghz),
        )
        metadata = {
            **self.metadata,
            **dict(target_metadata),
            "frequency_unit": "GHz",
            "feature_scaling": "standard" if self.feature_transformer.scale else "none",
            "feature_mean": self.feature_transformer.mean_.tolist()
            if self.feature_transformer.mean_ is not None
            else None,
            "feature_scale": self.feature_transformer.scale_.tolist()
            if self.feature_transformer.scale_ is not None
            else None,
        }
        return MLDataset(
            X=features.X,
            target=target.target,
            split_labels=split_labels,
            simulation_indices=features.simulation_indices,
            frequencies_ghz=features.frequencies_ghz,
            feature_names=features.feature_names,
            target_names=target.target_names,
            metadata=metadata,
        )

    def _save_if_requested(self, dataset: MLDataset, filename: str) -> None:
        if self.output_dir is not None:
            dataset.save(self.output_dir / filename)

    @staticmethod
    def _validate_row_alignment(
        features: PcbFeatureMatrix,
        target: TargetMatrix,
    ) -> None:
        if not np.array_equal(features.simulation_indices, target.simulation_indices):
            raise ValueError("Target simulation metadata does not match feature rows.")
        if not np.allclose(features.frequencies_ghz, target.frequencies_ghz):
            raise ValueError("Target frequency metadata does not match feature rows.")

    @staticmethod
    def _align_parameters(
        parameters: PcbParameters | pd.DataFrame,
        responses: SParameterDataset,
    ) -> pd.DataFrame:
        frame = parameters.dataframe if isinstance(parameters, PcbParameters) else parameters
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("parameters must be a PcbParameters or pandas DataFrame.")
        if "SIMU_INDEX" not in frame:
            raise ValueError("parameters must contain a SIMU_INDEX column.")

        aligned = frame.copy()
        raw_indices = aligned["SIMU_INDEX"].to_numpy(dtype=float)
        integer_indices = raw_indices.astype(np.int64)
        if not np.allclose(raw_indices, integer_indices):
            raise ValueError("SIMU_INDEX values must be integer-valued.")
        if len(np.unique(integer_indices)) != len(integer_indices):
            raise ValueError("SIMU_INDEX values must be unique.")

        aligned["SIMU_INDEX"] = integer_indices
        indexed = aligned.set_index("SIMU_INDEX", drop=False)
        requested = [int(index) for index in responses.simulation_indices]
        missing = [index for index in requested if index not in indexed.index]
        if missing:
            raise ValueError(
                "parameters are missing response SIMU_INDEX values: "
                + ", ".join(str(index) for index in missing)
            )
        return indexed.loc[requested].reset_index(drop=True)
