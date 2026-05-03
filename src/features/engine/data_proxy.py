"""
DataProxy - Expression composition engine
Allows composing alpha factor expressions using operator overloading.
"""

from __future__ import annotations
from typing import Union
import polars as pl


class DataProxy:
    """Feature data proxy for composing expressions"""

    def __init__(self, df: pl.DataFrame) -> None:
        self.name: str = df.columns[-1]
        self.df: pl.DataFrame = df.rename({self.name: "data"})

    def result(self, s: pl.Series) -> "DataProxy":
        result: pl.DataFrame = self.df[["datetime", "vt_symbol"]]
        result = result.with_columns(other=s)
        return DataProxy(result)

    def __add__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] + other.df["data"]
        else:
            s = self.df["data"] + other
        return self.result(s)

    def __sub__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] - other.df["data"]
        else:
            s = self.df["data"] - other
        return self.result(s)

    def __mul__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] * other.df["data"]
        else:
            s = self.df["data"] * other
        return self.result(s)

    def __rmul__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] * other.df["data"]
        else:
            s = self.df["data"] * other
        return self.result(s)

    def __truediv__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] / other.df["data"]
        else:
            s = self.df["data"] / other
        return self.result(s)

    def __abs__(self) -> "DataProxy":
        s: pl.Series = self.df["data"].abs()
        return self.result(s)

    def __gt__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] > other.df["data"]
        else:
            s = self.df["data"] > other
        return self.result(s.cast(pl.Int32))

    def __ge__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] >= other.df["data"]
        else:
            s = self.df["data"] >= other
        return self.result(s.cast(pl.Int32))

    def __lt__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] < other.df["data"]
        else:
            s = self.df["data"] < other
        return self.result(s.cast(pl.Int32))

    def __le__(self, other: Union["DataProxy", int, float]) -> "DataProxy":
        if isinstance(other, DataProxy):
            s: pl.Series = self.df["data"] <= other.df["data"]
        else:
            s = self.df["data"] <= other
        return self.result(s.cast(pl.Int32))

    def __eq__(self, other: Union["DataProxy", int, float]) -> "DataProxy":    # type: ignore
        if isinstance(other, DataProxy):
            s = self.df["data"] == other.df["data"]
        else:
            s = self.df["data"] == other
        return self.result(s.cast(pl.Int32))
