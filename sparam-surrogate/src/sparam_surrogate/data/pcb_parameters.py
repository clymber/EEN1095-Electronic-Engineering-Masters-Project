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
Data structure of PCB design parameter variations: parameter.csv
"""

from pathlib import Path
from typing import overload

import pandas as pd


# %%
class PcbParameters:
    """
    Wrapper for PCB parameter.csv data.

    This class stores the raw parameter table and provides project-specific
    utilities for inspection and physical validation.


    Important:
        This class is specifically designed for dataset
        *linkOn8CavityStackBetween10x10Array_19_08_2021*, with fixed columns:
        EPS, TAND, PITCH, TRACE_LEN, START, VIAR, ANTIPADR, TDIEL, DISTTL,
        TLWIDTH and SIMU_INDEX.
        It may not be applicable to other datasets without modification.
    """

    def __init__(self, data: pd.DataFrame | Path | str, copy: bool = True):
        if isinstance(data, pd.DataFrame):
            self._parameters = data.copy() if copy else data
        else:
            self._parameters = pd.read_csv(data, encoding="utf-8")

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        Return the underlying data frame of parameter table.
        """
        return self._parameters

    @overload
    def __getitem__(self, key: str) -> pd.Series: ...

    @overload
    def __getitem__(self, key: list[str] | pd.Index) -> pd.DataFrame: ...

    def __getitem__(self, key: str | list[str] | pd.Index) -> pd.Series | pd.DataFrame:
        """
        Select one or more parameter columns using DataFrame-style syntax.
        """
        return self._parameters[key]

    def assign_columns(self, **kw_args) -> "PcbParameters":
        """
        Assign new columns.
        Existing columns that are re-assigned will be overwritten.
        """
        self._parameters = self._parameters.assign(**kw_args)
        return self

    def _select_columns(
        self, columns: str | list[str] | pd.Index | None = None
    ) -> pd.DataFrame:
        """
        Return the full parameter table or a requested subset of columns.
        """
        if columns is None:
            return self._parameters
        if isinstance(columns, str):
            columns = [columns]
        return self._parameters.loc[:, list(columns)]

    def preview(self, columns: str | list[str] | pd.Index | None = None) -> str:
        """
        Return a compact string preview of the PCB parameter table.

        The preview shows a limited number of rows while preserving the
        selected columns. By default, all columns are included.
        """
        with pd.option_context(
            "display.max_rows",
            11,
            "display.min_rows",
            10,
            "display.max_columns",
            None,
            "display.width",
            120,
        ):
            return str(self._select_columns(columns))

    def structural_summary(
        self, columns: str | list[str] | pd.Index | None = None
    ) -> None:
        """
        Print a structural summary of selected PCB parameter columns.

        By default, all columns are included.
        """
        self._select_columns(columns).info()

    def statistical_summary(
        self, columns: str | list[str] | pd.Index | None = None
    ) -> None:
        """
        Print a statistical summary of selected PCB design parameters.

        Summary of statistics: mean, median, standard deviation, and range...
        The SIMU_INDEX column is excluded because it is an identifier rather
        than a physical design parameter. Remaining columns are described in
        groups of five to keep the printed output readable. By default, all
        columns are considered.
        """
        # Exclude SIMU_INDEX column since it's semantically not numerical
        df = self._select_columns(columns).drop(
            columns=["SIMU_INDEX"], errors="ignore"
        )

        descriptions = []
        for col in range(0, len(df.columns), 5):
            descriptions.append(str(df.iloc[:, col : col + 5].describe()))

        if descriptions:
            print("\n\n".join(descriptions))

    def check_geometry_relationship(self) -> int:
        """
        Check and report the validity of geometry relationships between columns.

        Some parameters should not be interpreted independently only. For
        example, the antipad radius is expected to be larger than the via
        radius.
        """

        fail_count = 0
        params = self._parameters
        constraints = {
            "ANTIPADR > VIAR": (
                lambda df: df["ANTIPADR"] > df["VIAR"],
                "ANTIPADR must be greater than VIAR",
            ),
            "DISTTL > TLWIDTH": (
                lambda df: df["DISTTL"] > df["TLWIDTH"],
                "DISTTL must be greater than TLWIDTH",
            ),
        }

        for name, (is_valid, message) in constraints.items():
            print(f"Checking constraint {name}:")

            invalid_mask = ~is_valid(params)
            invalid_records = params.loc[invalid_mask]

            if not invalid_records.empty:
                fail_count += len(invalid_records)
                print(f"Invalid geometry - {message}:\n{invalid_records}")
            else:
                print(f"\tChecked constraint {name} OK")

        return fail_count

    def check_physical_range(self) -> int:
        """
        Check that each parameter is within its physical range.
        Returns number of failures.
        """
        fail_count = 0
        params = self._parameters
        constraints = {
            "EPS": (lambda s: s > 1.0, "EPS must be greater than 1.0"),
            "TAND": (lambda s: s >= 0, "TAND must be non-negative"),
            "PITCH": (lambda s: s > 0, "PITCH must be positive"),
            "TRACE_LEN": (lambda s: s > 0, "TRACE_LEN must be positive"),
            "START": (lambda s: s >= 0, "START must be non-negative"),
            "VIAR": (lambda s: s > 0, "VIAR must be positive"),
            "ANTIPADR": (lambda s: s > 0, "ANTIPADR must be positive"),
            "TDIEL": (lambda s: s > 0, "TDIEL must be positive"),
            "DISTTL": (lambda s: s > 0, "DISTTL must be positive"),
            "TLWIDTH": (lambda s: s > 0, "TLWIDTH must be positive"),
        }

        for col, (cond, msg) in constraints.items():
            if col in params.columns:
                print(f"Checking column {col}:")

                invalid = params[~params[col].apply(cond)]
                if not invalid.empty:
                    fail_count += len(invalid)
                    print(f"Invalid values in {col}: {msg}")
                    print(invalid[[col]])
                else:
                    print(f"\tChecked column {col} OK")

        return fail_count
