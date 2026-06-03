"""
Target construction for scalar and full-S-matrix learning tasks.
"""

from dataclasses import dataclass

import numpy as np

from .s_parameter_dataset import SParameterDataset


@dataclass(frozen=True)
class TargetMatrix:
    """
    Target array and row metadata aligned to design-frequency feature rows.
    """

    target: np.ndarray  # Target rows, one per design-frequency pair.
    target_names: tuple[str, ...]  # Column names aligned with target.
    simulation_indices: np.ndarray  # Simulation ID aligned with each row.
    frequencies_ghz: np.ndarray  # Frequency value aligned with each row.


class TargetBuilder:
    """
    Build model targets from aligned S-parameter responses.
    """

    @staticmethod
    def build_scalar(
        responses: SParameterDataset,
        pair: tuple[int, int],
        representation: str = "db",
    ) -> TargetMatrix:
        """
        Build one selected scalar response target.

        Parameters
        ----------
        responses:
            Aligned S-parameter dataset.
        pair:
            One-based ``(receiver, source)`` port pair to extract.
        representation:
            Scalar target representation. Currently only ``"db"`` is supported
            because selected paths are stored in ``responses.through_s_db``.
        """
        if representation.lower() != "db":
            raise ValueError(
                "Only the 'db' scalar representation is currently supported."
            )

        normalized_pair = (int(pair[0]), int(pair[1]))
        if normalized_pair not in responses.port_pairs:
            raise ValueError(
                f"Scalar port pair {normalized_pair} is not available in responses."
            )

        pair_index = responses.port_pairs.index(normalized_pair)
        target = responses.through_s_db[:, :, pair_index].reshape(-1, 1)
        return TargetMatrix(
            target=target,
            target_names=(responses.response_column_name(normalized_pair),),
            simulation_indices=TargetBuilder._expanded_simulation_indices(responses),
            frequencies_ghz=TargetBuilder._expanded_frequencies(responses),
        )

    @staticmethod
    def build_full_smatrix(responses: SParameterDataset) -> TargetMatrix:
        """
        Build a flattened full complex S-matrix target.

        The output order is row-major over the parsed S-matrix. For each
        ``S(receiver, source)`` entry, real and imaginary components are adjacent:
        ``REAL_S1_1, IMAG_S1_1, REAL_S1_2, IMAG_S1_2, ...``.
        """
        full_s_matrix = responses.full_s_matrix
        n_ports = full_s_matrix.shape[2]
        flattened_matrices = full_s_matrix.reshape(-1, n_ports, n_ports)

        columns: list[np.ndarray] = []
        target_names: list[str] = []
        for receiver in range(n_ports):
            for source in range(n_ports):
                values = flattened_matrices[:, receiver, source]
                receiver_label = receiver + 1
                source_label = source + 1
                target_names.extend(
                    [
                        f"REAL_S{receiver_label}_{source_label}",
                        f"IMAG_S{receiver_label}_{source_label}",
                    ]
                )
                columns.extend([values.real, values.imag])

        return TargetMatrix(
            target=np.column_stack(columns),
            target_names=tuple(target_names),
            simulation_indices=TargetBuilder._expanded_simulation_indices(responses),
            frequencies_ghz=TargetBuilder._expanded_frequencies(responses),
        )

    @staticmethod
    def _expanded_simulation_indices(responses: SParameterDataset) -> np.ndarray:
        """
        Return design-major simulation metadata for target rows.
        """
        return np.repeat(responses.simulation_indices, len(responses.frequencies_ghz))

    @staticmethod
    def _expanded_frequencies(responses: SParameterDataset) -> np.ndarray:
        """
        Return design-major frequency metadata for target rows.
        """
        return np.tile(responses.frequencies_ghz, len(responses.simulation_indices))
