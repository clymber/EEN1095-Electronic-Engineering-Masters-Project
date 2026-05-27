# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python (sparam-surrogate)
#     language: python
#     name: sparam-surrogate
# ---

# %%
"""
Exploratory Data Analysis for PCB Dataset.
"""

from typing import overload

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from .pcb_parameters import PcbParameters
from .s_parameter_dataset import SParameterDataset


# %%
class PcbDatasetEDA:
    """
    Derived features and exploratory plots for PCB parameters.
    """

    def __init__(
        self,
        parameters: PcbParameters,
        responses: SParameterDataset | None = None,
    ) -> None:
        features = parameters.dataframe.copy()
        features = features.assign(
            BOARD_HEIGHT=2 * features["START"] + 9 * features["PITCH"],
            BOARD_WIDTH=(
                2 * features["START"] + features["TRACE_LEN"] + 18 * features["PITCH"]
            ),
            ANTIPAD_TO_VIA_RATIO=features["ANTIPADR"] / features["VIAR"],
            TLWIDTH_TO_DIEL_RATIO=features["TLWIDTH"] / features["TDIEL"],
            TRACE_ASPECT_RATIO=features["TRACE_LEN"] / features["PITCH"],
        )
        self._features = features.assign(
            BOARD_AREA=features["BOARD_HEIGHT"] * features["BOARD_WIDTH"],
        )
        self._responses = responses

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        Return the underlying data frame of parameter table.
        """
        return self._features

    @property
    def responses(self) -> SParameterDataset | None:
        """
        Return the optional aligned frequency-response dataset.
        """
        return self._responses

    @overload
    def __getitem__(self, key: str) -> pd.Series: ...

    @overload
    def __getitem__(self, key: list[str] | pd.Index) -> pd.DataFrame: ...

    def __getitem__(self, key: str | list[str] | pd.Index) -> pd.Series | pd.DataFrame:
        """
        Select one or more parameter columns using DataFrame-style syntax.
        """
        return self._features[key]

    def plot_board_geometry_verification(
        self,
        feature_pairs: list[tuple[str, str]] | None = None,
        *,
        show: bool = True,
    ) -> Figure | None:
        """
        Plot the raw dimensions used to construct board geometry features.
        """
        if feature_pairs is None:
            feature_pairs = [
                ("PITCH", "BOARD_HEIGHT"),
                ("TRACE_LEN", "BOARD_WIDTH"),
            ]
        if not feature_pairs:
            return None

        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(feature_pairs),
            figsize=(6 * len(feature_pairs), 4),
            constrained_layout=True,
            squeeze=False,
        )

        for ax, (x_feature, y_feature) in zip(
            axes.ravel(), feature_pairs, strict=False
        ):
            ax.scatter(
                self._features[x_feature],
                self._features[y_feature],
                s=8,
                alpha=0.35,
            )
            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            ax.set_title(f"{y_feature} vs {x_feature}")

        fig.suptitle("Derived board-geometry verification", fontsize=14)

        if show:
            plt.show()

        return fig

    def plot_ratio_relationships(
        self,
        parameter_pairs: list[tuple[str, str, str]] | None = None,
        *,
        show: bool = True,
    ) -> Figure | None:
        """
        Plot physical parameter pairs colored by their derived ratios.
        """
        if parameter_pairs is None:
            parameter_pairs = [
                ("VIAR", "ANTIPADR", "ANTIPAD_TO_VIA_RATIO"),
                ("TDIEL", "TLWIDTH", "TLWIDTH_TO_DIEL_RATIO"),
                ("PITCH", "TRACE_LEN", "TRACE_ASPECT_RATIO"),
            ]
        if not parameter_pairs:
            return None

        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(parameter_pairs),
            figsize=(5 * len(parameter_pairs), 4),
            constrained_layout=True,
            squeeze=False,
        )

        for ax, (x_feature, y_feature, color_feature) in zip(
            axes.ravel(), parameter_pairs, strict=False
        ):
            scatter = ax.scatter(
                self._features[x_feature],
                self._features[y_feature],
                c=self._features[color_feature],
                s=8,
                alpha=0.6,
            )
            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            ax.set_title(f"{y_feature} vs {x_feature}")
            fig.colorbar(scatter, ax=ax, label=color_feature)

        if show:
            plt.show()

        return fig

    def _select_columns(
        self, columns: str | list[str] | pd.Index | None = None
    ) -> pd.DataFrame:
        """
        Return the full parameter table or a requested subset of columns.
        """
        if columns is None:
            return self._features
        if isinstance(columns, str):
            columns = [columns]
        return self._features.loc[:, list(columns)]

    def statistical_summary(
        self, columns: str | list[str] | pd.Index | None = None
    ) -> None:
        """
        Print statistical summaries for selected analysis-table columns.
        """
        selected = self._select_columns(columns).drop(
            columns=["SIMU_INDEX"], errors="ignore"
        )
        descriptions = [
            str(selected.iloc[:, col : col + 5].describe())
            for col in range(0, len(selected.columns), 5)
        ]
        if descriptions:
            print("\n\n".join(descriptions))

    def plot_distribution_histograms(
        self,
        features: list[str] | pd.Index | None = None,
        *,
        show: bool = True,
    ) -> Figure | None:
        """
        Plot histograms for selected raw or derived parameter columns.
        """
        if features is None:
            features = self._features.columns.to_list()
        else:
            features = list(features)
        if not features:
            return None

        n_columns = 2
        n_rows = (len(features) + 1) // n_columns
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_columns,
            figsize=(12, 14),
            constrained_layout=True,
            squeeze=False,
        )
        axes = axes.ravel()

        for ax, feature in zip(axes, features, strict=False):
            ax.hist(self._features[feature], bins=30)
            ax.set_title(feature)
            ax.set_xlabel(feature)
            ax.set_ylabel("Count")

        for ax in axes[len(features) :]:
            ax.set_visible(False)

        if show:
            plt.show()

        return fig

    def plot_correlation_heatmap(
        self,
        features: list[str] | pd.Index | None = None,
        *,
        show: bool = True,
    ) -> Figure | None:
        """
        Plot a correlation heatmap for selected analysis-table columns.
        """
        if features is None:
            selected = self._features
        else:
            features = list(features)
            if not features:
                return None
            selected = self._features[features]

        corr = selected.corr(numeric_only=True)
        if corr.empty:
            return None

        feature_labels = corr.columns.to_list()
        fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")

        ax.set_xticks(range(len(feature_labels)))
        ax.set_yticks(range(len(feature_labels)))
        ax.set_xticklabels(feature_labels, rotation=90)
        ax.set_yticklabels(feature_labels)
        ax.set_title("Correlation heatmap of physical and derived parameters")

        for row_idx in range(len(feature_labels)):
            for col_idx in range(len(feature_labels)):
                value = corr.iloc[row_idx, col_idx]
                if row_idx == col_idx or abs(value) >= 0.5:
                    ax.text(
                        col_idx,
                        row_idx,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )

        fig.colorbar(im, ax=ax, label="Pearson correlation coefficient")

        if show:
            plt.show()

        return fig

    def plot_physical_relationships(
        self,
        feature_pairs: list[tuple[str, str]] | None = None,
        *,
        show: bool = True,
    ) -> Figure | None:
        """
        Plot engineering-relevant relationships between physical parameters.
        """
        if feature_pairs is None:
            feature_pairs = [
                ("VIAR", "ANTIPADR"),
                ("TLWIDTH", "TDIEL"),
                ("PITCH", "TRACE_LEN"),
                ("DISTTL", "TLWIDTH"),
            ]
        if not feature_pairs:
            return None

        constraint_labels = {
            ("VIAR", "ANTIPADR"): "ANTIPADR = VIAR",
            ("DISTTL", "TLWIDTH"): "TLWIDTH = DISTTL",
        }
        n_columns = 2
        n_rows = (len(feature_pairs) + 1) // n_columns
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_columns,
            figsize=(12, 4 * n_rows),
            constrained_layout=True,
            squeeze=False,
        )
        axes = axes.ravel()

        for ax, (x_feature, y_feature) in zip(axes, feature_pairs, strict=False):
            ax.scatter(
                self._features[x_feature],
                self._features[y_feature],
                s=8,
                alpha=0.35,
            )

            label = constraint_labels.get((x_feature, y_feature))
            if label is not None:
                lower = min(
                    self._features[x_feature].min(),
                    self._features[y_feature].min(),
                )
                upper = max(
                    self._features[x_feature].max(),
                    self._features[y_feature].max(),
                )
                ax.plot(
                    [lower, upper],
                    [lower, upper],
                    linestyle="--",
                    linewidth=1,
                )
                ax.text(lower, lower, label, fontsize=8, va="bottom")

            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            ax.set_title(f"{y_feature} vs {x_feature}")

        for ax in axes[len(feature_pairs) :]:
            ax.set_visible(False)

        fig.suptitle("Physical parameter relationships", fontsize=14)

        if show:
            plt.show()

        return fig

    def correlation_pairs(
        self,
        features: list[str] | pd.Index | None = None,
    ) -> pd.DataFrame:
        """
        Return distinct Pearson correlation pairs ordered by magnitude.
        """
        if features is None:
            selected = self._features
        else:
            selected = self._features[list(features)]

        corr = selected.corr(numeric_only=True)
        mask_upper_triangle = pd.DataFrame(
            data=False, index=corr.index, columns=corr.columns
        )

        for row_idx, feature_1 in enumerate(corr.index):
            for col_idx, feature_2 in enumerate(corr.columns):
                if row_idx < col_idx:
                    mask_upper_triangle.loc[feature_1, feature_2] = True

        corr_pairs = (
            corr.where(mask_upper_triangle)
            .stack()
            .rename("corr")
            .reset_index()
            .rename(
                columns={
                    "level_0": "feature_1",
                    "level_1": "feature_2",
                }
            )
        )
        corr_pairs["abs_corr"] = corr_pairs["corr"].abs()

        return corr_pairs.sort_values("abs_corr", ascending=False)

    def response_frame_at_frequency(self, frequency_ghz: float = 10.0) -> pd.DataFrame:
        """
        Join parameter/derived features to aligned responses at one frequency.
        """
        responses = self._require_responses()
        if "SIMU_INDEX" not in self._features.columns:
            raise ValueError(
                "Response analysis requires a SIMU_INDEX column in PCB parameters."
            )

        features = self._features.copy()
        numeric_indices = pd.to_numeric(features["SIMU_INDEX"], errors="coerce")
        if numeric_indices.isna().any() or not np.allclose(
            numeric_indices.to_numpy(),
            numeric_indices.astype(np.int64).to_numpy(),
        ):
            raise ValueError("SIMU_INDEX values must be integer-valued.")
        features["SIMU_INDEX"] = numeric_indices.astype(np.int64)
        if features["SIMU_INDEX"].duplicated().any():
            raise ValueError("SIMU_INDEX values must be unique.")

        return features.merge(
            responses.at_frequency(frequency_ghz),
            on="SIMU_INDEX",
            how="inner",
            validate="one_to_one",
        )

    def response_correlation_pairs(
        self,
        frequency_ghz: float = 10.0,
        features: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Return feature-to-response Pearson correlations ordered by magnitude.
        """
        frame = self.response_frame_at_frequency(frequency_ghz)
        response_columns = [
            SParameterDataset.response_column(pair)
            for pair in self._require_responses().port_pairs
        ]
        if features is None:
            feature_columns = [
                column
                for column in frame.select_dtypes(include="number").columns
                if column not in [*response_columns, "SIMU_INDEX"]
            ]
        else:
            feature_columns = (
                frame.loc[:, features].select_dtypes(include="number").columns.to_list()
            )

        correlations = [
            {
                "feature": feature,
                "response": response,
                "corr": frame[feature].corr(frame[response]),
            }
            for feature in feature_columns
            for response in response_columns
        ]
        result = pd.DataFrame(correlations, columns=["feature", "response", "corr"])
        if result.empty:
            return result.assign(abs_corr=pd.Series(dtype=float))
        result["abs_corr"] = result["corr"].abs()
        return result.sort_values(
            "abs_corr", ascending=False, na_position="last"
        ).reset_index(drop=True)

    def plot_response_distributions(
        self,
        frequency_ghz: float = 10.0,
        *,
        show: bool = True,
    ) -> Figure | None:
        """
        Plot selected response distributions at one frequency.
        """
        response_frame = self._require_responses().at_frequency(frequency_ghz)
        response_columns = response_frame.columns.drop("SIMU_INDEX").to_list()
        if not response_columns:
            return None

        n_columns = 2
        n_rows = (len(response_columns) + 1) // n_columns
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_columns,
            figsize=(12, 4 * n_rows),
            constrained_layout=True,
            squeeze=False,
        )
        axes = axes.ravel()
        for ax, response in zip(axes, response_columns, strict=False):
            ax.hist(response_frame[response], bins=30)
            ax.set_title(f"{response} at {frequency_ghz:g} GHz")
            ax.set_xlabel(f"{response} (dB)")
            ax.set_ylabel("Count")

        for ax in axes[len(response_columns) :]:
            ax.set_visible(False)

        fig.suptitle("Through-path response distributions", fontsize=14)
        if show:
            plt.show()
        return fig

    def plot_parameter_response_relationships(
        self,
        pair: tuple[int, int] = (7, 1),
        frequency_ghz: float = 10.0,
        features: list[str] | None = None,
        *,
        show: bool = True,
    ) -> Figure | None:
        """
        Plot selected PCB features against one transmission response target.
        """
        responses = self._require_responses()
        if pair not in responses.port_pairs:
            raise ValueError(f"Response port pair {pair} was not extracted.")
        target = SParameterDataset.response_column(pair)
        frame = self.response_frame_at_frequency(frequency_ghz)
        if features is None:
            features = [
                feature
                for feature in ["TRACE_LEN", "TAND", "EPS", "TRACE_ASPECT_RATIO"]
                if feature in frame.columns
            ]
        if not features:
            return None

        n_columns = 2
        n_rows = (len(features) + 1) // n_columns
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_columns,
            figsize=(12, 4 * n_rows),
            constrained_layout=True,
            squeeze=False,
        )
        axes = axes.ravel()
        for ax, feature in zip(axes, features, strict=False):
            ax.scatter(frame[feature], frame[target], s=8, alpha=0.35)
            ax.set_xlabel(feature)
            ax.set_ylabel(f"{target} (dB)")
            ax.set_title(f"{target} vs {feature}")

        for ax in axes[len(features) :]:
            ax.set_visible(False)

        fig.suptitle(f"Response relationships at {frequency_ghz:g} GHz", fontsize=14)
        if show:
            plt.show()
        return fig

    def plot_through_response_curves(
        self,
        pairs: list[tuple[int, int]] | None = None,
        simulation_indices: list[int] | None = None,
        *,
        show: bool = True,
    ) -> Figure | None:
        """
        Plot median response curves with percentile bands and optional overlays.
        """
        responses = self._require_responses()
        if pairs is None:
            pairs = list(responses.port_pairs)
        if not pairs:
            return None
        unknown_pairs = [pair for pair in pairs if pair not in responses.port_pairs]
        if unknown_pairs:
            raise ValueError(
                f"Response port pairs were not extracted: {unknown_pairs}."
            )

        overlay_rows: list[int] = []
        if simulation_indices is not None:
            row_by_index = {
                int(index): row
                for row, index in enumerate(responses.simulation_indices)
            }
            unknown_indices = [
                index for index in simulation_indices if index not in row_by_index
            ]
            if unknown_indices:
                raise ValueError(
                    "Simulation indices have no extracted responses: "
                    f"{unknown_indices}."
                )
            overlay_rows = [row_by_index[index] for index in simulation_indices]

        n_columns = 2
        n_rows = (len(pairs) + 1) // n_columns
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_columns,
            figsize=(12, 4 * n_rows),
            constrained_layout=True,
            squeeze=False,
        )
        axes = axes.ravel()
        pair_columns = {
            pair: column for column, pair in enumerate(responses.port_pairs)
        }
        for ax, pair in zip(axes, pairs, strict=False):
            curves = responses.through_s_db[:, :, pair_columns[pair]]
            lower, median, upper = np.percentile(curves, [10, 50, 90], axis=0)
            ax.plot(responses.frequencies_ghz, median, label="Median")
            ax.fill_between(
                responses.frequencies_ghz,
                lower,
                upper,
                alpha=0.2,
                label="10th-90th percentile",
            )
            for row in overlay_rows:
                simulation_index = int(responses.simulation_indices[row])
                ax.plot(
                    responses.frequencies_ghz,
                    curves[row],
                    linewidth=1,
                    alpha=0.8,
                    label=f"SIMU_INDEX {simulation_index}",
                )
            response = SParameterDataset.response_column(pair)
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel(f"{response} (dB)")
            ax.set_title(response)
            ax.legend(fontsize=8)

        for ax in axes[len(pairs) :]:
            ax.set_visible(False)

        fig.suptitle("Through-path response curves", fontsize=14)
        if show:
            plt.show()
        return fig

    def _require_responses(self) -> SParameterDataset:
        if self._responses is None:
            raise RuntimeError(
                "Response analysis requires an SParameterDataset instance."
            )
        return self._responses


# %%
