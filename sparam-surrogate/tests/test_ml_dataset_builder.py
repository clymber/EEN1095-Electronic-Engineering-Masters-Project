"""
Integration tests for :class:`MLDatasetBuilder`.
"""

import numpy as np
import pandas as pd

from sparam_surrogate.data import (
    DesignFrequencySplitter,
    MLDataset,
    MLDatasetBuilder,
    PcbFeatureTransformer,
    PcbParameters,
    SParameterDataset,
)


def _parameters() -> PcbParameters:
    """
    Return shuffled parameter rows so the builder must align by SIMU_INDEX.
    """
    return PcbParameters(
        pd.DataFrame(
            {
                "SIMU_INDEX": [30, 10, 50, 20, 40],
                "EPS": [3.0, 1.0, 5.0, 2.0, 4.0],
            }
        )
    )


def _responses() -> SParameterDataset:
    """
    Return aligned response data for five designs and two frequencies.
    """
    matrices = np.zeros((5, 2, 2, 2), dtype=complex)
    for design in range(5):
        for frequency in range(2):
            base = 100 * design + 10 * frequency
            matrices[design, frequency] = np.array(
                [
                    [base + 1 + 0.1j, base + 2 + 0.2j],
                    [base + 3 + 0.3j, base + 4 + 0.4j],
                ],
                dtype=complex,
            )

    return SParameterDataset(
        simulation_indices=[10, 20, 30, 40, 50],
        frequencies_ghz=[1.0, 2.0],
        port_pairs=[(2, 1)],
        through_s_db=np.array(
            [
                [[1.0], [2.0]],
                [[3.0], [4.0]],
                [[5.0], [6.0]],
                [[7.0], [8.0]],
                [[9.0], [10.0]],
            ],
            dtype=float,
        ),
        full_s_matrix=matrices,
    )


def _builder(output_dir=None) -> MLDatasetBuilder:
    """
    Return a deterministic builder with unscaled one-feature inputs.
    """
    return MLDatasetBuilder(
        splitter=DesignFrequencySplitter(
            test_size=0.2,
            val_size=0.2,
            random_state=123,
        ),
        feature_transformer=PcbFeatureTransformer(
            feature_columns=["EPS"],
            scale=False,
        ),
        output_dir=output_dir,
    )


class TestMLDatasetBuilder:
    """
    Tests for end-to-end ML dataset assembly.
    """

    def test_builds_scalar_and_full_smatrix_datasets_with_identical_inputs(
        self,
    ) -> None:
        """
        Scalar and full-S-matrix datasets share the same feature matrix.
        """
        builder = _builder()

        scalar = builder.build_scalar_dataset(_parameters(), _responses(), pair=(2, 1))
        full = builder.build_full_smatrix_dataset(_parameters(), _responses())

        np.testing.assert_allclose(scalar.X, full.X)
        np.testing.assert_array_equal(scalar.split_labels, full.split_labels)
        np.testing.assert_array_equal(scalar.simulation_indices, full.simulation_indices)
        np.testing.assert_allclose(scalar.frequencies_ghz, full.frequencies_ghz)
        assert scalar.target.shape == (10, 1)
        assert full.target.shape == (10, 8)
        np.testing.assert_allclose(
            scalar.X[:, 0],
            [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
        )
        np.testing.assert_allclose(
            scalar.X[:, 1],
            [1.0, 2.0] * 5,
        )

    def test_split_labels_have_no_simulation_leakage(self) -> None:
        """
        Every SIMU_INDEX appears in one split label only after frequency expansion.
        """
        dataset = _builder().build_scalar_dataset(
            _parameters(),
            _responses(),
            pair=(2, 1),
        )

        label_by_simulation: dict[int, set[str]] = {}
        for simulation_index, split_label in zip(
            dataset.simulation_indices,
            dataset.split_labels,
            strict=True,
        ):
            label_by_simulation.setdefault(int(simulation_index), set()).add(
                str(split_label)
            )

        assert set(dataset.split_labels.tolist()) == {"train", "val", "test"}
        assert all(len(labels) == 1 for labels in label_by_simulation.values())

    def test_writes_processed_files_and_reloads_them(self, tmp_path) -> None:
        """
        Builders with an output directory save reloadable processed datasets.
        """
        builder = _builder(output_dir=tmp_path)

        scalar = builder.build_scalar_dataset(_parameters(), _responses(), pair=(2, 1))
        full = builder.build_full_smatrix_dataset(_parameters(), _responses())

        scalar_path = tmp_path / "scalar_baseline_dataset.npz"
        full_path = tmp_path / "full_smatrix_dataset.npz"
        loaded_scalar = MLDataset.load(scalar_path)
        loaded_full = MLDataset.load(full_path)

        assert scalar_path.is_file()
        assert full_path.is_file()
        np.testing.assert_allclose(loaded_scalar.X, scalar.X)
        np.testing.assert_allclose(loaded_scalar.target, scalar.target)
        np.testing.assert_allclose(loaded_full.X, full.X)
        np.testing.assert_allclose(loaded_full.target, full.target)
        np.testing.assert_array_equal(loaded_scalar.split_labels, scalar.split_labels)
        np.testing.assert_array_equal(loaded_full.simulation_indices, full.simulation_indices)
