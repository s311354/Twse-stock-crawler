"""Pandas value boundaries that never finalize/copy financial metadata."""
from __future__ import annotations

import pandas as pd


def metadata_free_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Copy tabular values, index and dtypes, but not DataFrame.attrs.

    Constructing from a DataFrame uses pandas' public constructor, rather than
    copy()/slicing followed by attrs.clear(): those operations have already
    deep-copied attrs. A value copy also protects callers on pandas without CoW.
    This boundary is for scalar market/annual tables, not object-valued filings.
    """
    return pd.DataFrame(data, copy=True)


SCREEN_SOURCE_ROWS_ATTR = "screen_source_rows"
