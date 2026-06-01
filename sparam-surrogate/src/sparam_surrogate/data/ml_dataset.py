"""
Model-ready arrays and metadata for surrogate training.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


class MLDataset:
    """
    Container for model-ready features, targets, splits, and row metadata.
    """

    def __init__(
        self,
        X: Sequence[Sequence[float]] | np.ndarray,
        target: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
        split_labels: Sequence[str] | np.ndarray,
        simulation_indices: Sequence[int] | np.ndarray,
        frequencies_ghz: Sequence[float] | np.ndarray,
        feature_names: Sequence[str],
        target_names: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Create a model-ready dataset.

        Parameters
        ----------
        X:
            Two-dimensional feature matrix with shape
            ``(n_rows, n_features)``. In this project each row is expected to
            represent one design-frequency sample, for example
            ``[geometric/material parameters, frequency]``.
        target:
            Target values aligned row-for-row with ``X``. Scalar targets may be
            passed as a one-dimensional sequence with shape ``(n_rows,)`` and
            are stored internally as ``(n_rows, 1)``. Multi-output targets, such
            as flattened full S-matrices, should use shape
            ``(n_rows, n_targets)``.
        split_labels:
            One-dimensional split label for each row of ``X``, usually values
            such as ``"train"``, ``"val"``, or ``"test"``. These labels must be
            produced from a split by ``SIMU_INDEX`` before frequency expansion.
        simulation_indices:
            One-dimensional source ``SIMU_INDEX`` metadata for each row of
            ``X``. Repeated values are expected after a design is expanded over
            multiple frequencies.
        frequencies_ghz:
            One-dimensional frequency metadata for each row of ``X``, in GHz.
            Its order must match the row order of ``X`` and ``target``.
        feature_names:
            Feature names in the same order as the columns of ``X``. The number
            of names must equal ``n_features``.
        target_names:
            Target names in the same order as the columns of ``target``. The
            number of names must equal ``n_targets`` after scalar targets are
            reshaped to two dimensions.
        metadata:
            Optional JSON-serializable dataset-level metadata, such as target
            mode, scaling information, source dataset name, or preprocessing
            settings.

        Raises
        ------
        ValueError
            If array dimensions are invalid, row metadata does not match
            ``X`` row count, feature/target names do not match column counts,
            numeric arrays contain non-finite values, or metadata is not
            JSON-serializable.
        """
        self._X = np.asarray(X, dtype=float)
        self._target = np.asarray(target, dtype=float)
        if self._target.ndim == 1:
            self._target = self._target.reshape(-1, 1)
        self._split_labels = np.asarray(split_labels, dtype=str)
        self._simulation_indices = np.asarray(simulation_indices, dtype=np.int64)
        self._frequencies_ghz = np.asarray(frequencies_ghz, dtype=float)
        self._feature_names = tuple(str(name) for name in feature_names)
        self._target_names = tuple(str(name) for name in target_names)
        self._metadata = dict(metadata or {})
        self._validate()

    @property
    def X(self) -> np.ndarray:
        """
        Return the feature matrix.
        """
        return self._X

    @property
    def target(self) -> np.ndarray:
        """
        Return the target matrix.
        """
        return self._target

    @property
    def split_labels(self) -> np.ndarray:
        """
        Return train/validation/test labels for each row.
        """
        return self._split_labels

    @property
    def simulation_indices(self) -> np.ndarray:
        """
        Return the source simulation index for each row.
        """
        return self._simulation_indices

    @property
    def frequencies_ghz(self) -> np.ndarray:
        """
        Return the frequency value for each row.
        """
        return self._frequencies_ghz

    @property
    def feature_names(self) -> tuple[str, ...]:
        """
        Return feature names in feature-matrix column order.
        """
        return self._feature_names

    @property
    def target_names(self) -> tuple[str, ...]:
        """
        Return target names in target-matrix column order.
        """
        return self._target_names

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Return additional JSON-serializable dataset metadata.
        """
        return dict(self._metadata)

    def save(self, path: Path | str) -> None:
        """
        Save arrays and metadata to a compressed NumPy archive.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as dataset_file:
            np.savez_compressed(
                dataset_file,
                X=self._X,
                target=self._target,
                split_labels=self._split_labels,
                simulation_indices=self._simulation_indices,
                frequencies_ghz=self._frequencies_ghz,
                feature_names=np.asarray(self._feature_names, dtype=str),
                target_names=np.asarray(self._target_names, dtype=str),
                metadata_json=np.asarray(self._metadata_json()),
            )

    @classmethod
    def load(cls, path: Path | str) -> "MLDataset":
        """
        Load a dataset saved by :meth:`save`.
        """
        with np.load(path, allow_pickle=False) as archive:
            metadata_json = str(np.asarray(archive["metadata_json"]).item())
            return cls(
                X=archive["X"],
                target=archive["target"],
                split_labels=archive["split_labels"],
                simulation_indices=archive["simulation_indices"],
                frequencies_ghz=archive["frequencies_ghz"],
                feature_names=archive["feature_names"].tolist(),
                target_names=archive["target_names"].tolist(),
                metadata=json.loads(metadata_json),
            )

    def _metadata_json(self) -> str:
        try:
            return json.dumps(self._metadata, sort_keys=True)
        except TypeError as exc:
            raise ValueError("Metadata must be JSON-serializable.") from exc

    def _validate(self) -> None:
        if self._X.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix.")
        if self._target.ndim != 2:
            raise ValueError("Target must be a one- or two-dimensional array.")
        if len(self._X) == 0:
            raise ValueError("MLDataset must contain at least one row.")
        if not np.isfinite(self._X).all():
            raise ValueError("X contains non-finite values.")
        if not np.isfinite(self._target).all():
            raise ValueError("Target contains non-finite values.")

        row_count = self._X.shape[0]
        if self._target.shape[0] != row_count:
            raise ValueError(
                f"target rows ({self._target.shape[0]}) do not match "
                f"X rows ({row_count})."
            )
        if self._split_labels.shape != (row_count,):
            raise ValueError("Number of split labels must match X rows.")
        if self._simulation_indices.shape != (row_count,):
            raise ValueError("Number of simulation indices must match X rows.")
        if self._frequencies_ghz.shape != (row_count,):
            raise ValueError("Number of frequency metadata values must match X rows.")
        if not np.isfinite(self._frequencies_ghz).all():
            raise ValueError("Frequency metadata contains non-finite values.")
        if len(self._feature_names) != self._X.shape[1]:
            raise ValueError("Number of feature names must match X columns.")
        if len(self._target_names) != self._target.shape[1]:
            raise ValueError("Number of target names must match target columns.")
        if any(not name for name in self._feature_names):
            raise ValueError("Feature names must be non-empty.")
        if any(not name for name in self._target_names):
            raise ValueError("Target names must be non-empty.")
        self._metadata_json()
