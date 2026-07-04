"""
Base interfaces for neural S-parameter surrogate models.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from sparam_surrogate.models.base import SparamModel


class NeuralModel(SparamModel):
    """
    Common interface for neural S-parameter surrogate models.
    """

    def plot_training_history(self) -> Figure:
        """
        Plot scaled-unit training and validation MSE histories.
        """
        history = getattr(self, "history", None)
        if history is None:
            raise RuntimeError(f"{self.name} has no recorded training history.")

        history_data: dict[str, Any] = history.history
        history_frame = pd.DataFrame(history_data)
        best_epoch = int(history_frame["val_loss"].idxmin()) + 1
        best_val_loss = float(history_frame["val_loss"].min())
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(history_frame.index + 1, history_frame["loss"], label="train loss")
        ax.plot(history_frame.index + 1, history_frame["val_loss"], label="val loss")
        ax.axvline(
            best_epoch,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
        )
        ax.scatter(
            [best_epoch],
            [best_val_loss],
            color="black",
            s=30,
            zorder=3,
            label=f"best val epoch {best_epoch}",
        )
        ax.set_title(f"{self.model_name()} Training History")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss (scaled target units)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig
