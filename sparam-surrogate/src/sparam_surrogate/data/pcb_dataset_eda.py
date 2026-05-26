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

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from .pcb_parameters import PcbParameters


# %%
class PcbDatasetEDA:
    """
    Derived features and exploratory plots for PCB parameters.
    """

    def __init__(self, parameters: PcbParameters) -> None:
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

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        Return the underlying data frame of parameter table.
        """
        return self._features

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


# %%
