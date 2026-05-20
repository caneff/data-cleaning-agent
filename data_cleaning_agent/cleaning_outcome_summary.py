"""Verified before/after facts for the cleaning Streamlit UI (no Streamlit)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_dtype_equal

import data_cleaning_agent.source_row_identity as source_row_identity
import data_cleaning_agent.utils as utils

DEFAULT_NULL_TOP_K = 10
_ROW_ID_LABEL = "Source Row Identity"


def _column_heading(name: str, *, row_id_col: str | None = None) -> str:
    identity_label = row_id_col or source_row_identity.DEFAULT_SOURCE_ROW_IDENTITY_LABEL
    if str(name).strip() == identity_label:
        return _ROW_ID_LABEL
    return str(name)


def _same_values(left: pd.Series, right: pd.Series) -> bool:
    """True when values match; ignores dtype-only drift (e.g. int64 vs int32)."""
    if len(left) != len(right):
        return False
    a = left.reset_index(drop=True)
    b = right.reset_index(drop=True)
    if a.equals(b):
        return True
    try:
        return bool(np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True))
    except (TypeError, ValueError):
        return False


def _value_changes_by_row_id(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    *,
    row_id_col: str,
) -> list[dict[str, Any]]:
    """Count changed values on shared Source Row Identity values."""
    if row_id_col not in df_before.columns or row_id_col not in df_after.columns:
        return []

    before_ids = utils.first_column_as_series(df_before, row_id_col)
    after_ids = utils.first_column_as_series(df_after, row_id_col)
    after_id_set = set(after_ids.tolist())
    shared_ids = [value for value in before_ids.tolist() if value in after_id_set]
    if not shared_ids:
        return []

    common = sorted((set(df_before.columns) & set(df_after.columns)) - {row_id_col})
    changes: list[dict[str, Any]] = []
    for name in common:
        before = pd.DataFrame({
            row_id_col: before_ids,
            name: utils.first_column_as_series(df_before, name),
        })
        after = pd.DataFrame({
            row_id_col: after_ids,
            name: utils.first_column_as_series(df_after, name),
        })
        before = before.drop_duplicates(row_id_col).set_index(row_id_col)
        after = after.drop_duplicates(row_id_col).set_index(row_id_col)
        left = before.loc[shared_ids, name].reset_index(drop=True)
        right = after.loc[shared_ids, name].reset_index(drop=True)
        same = left.eq(right) | (left.isna() & right.isna())
        changed = int((~same.fillna(False)).sum())
        if changed:
            changes.append({"column": name, "changed_values": changed})
    changes.sort(key=lambda row: row["changed_values"], reverse=True)
    return changes


def build_cleaning_outcome_facts(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    *,
    row_id_col: str,
    null_top_k: int = DEFAULT_NULL_TOP_K,
) -> dict[str, Any]:
    """Build JSON-serializable facts comparing ``df_before`` to ``df_after``.

    Parameters
    ----------
    df_before, df_after
        Input and output of the same cleaner run.
    row_id_col
        Source Row Identity label selected by the Cleaning Run policy.
    null_top_k
        Max number of shared columns to list in ``null_deltas`` by absolute
        change in raw ``isna()`` count.
    """
    if null_top_k < 1:
        raise ValueError("null_top_k must be at least 1")

    before_cols = set(df_before.columns) - {row_id_col}
    after_cols = set(df_after.columns) - {row_id_col}
    dropped = sorted(before_cols - after_cols)
    added = sorted(after_cols - before_cols)

    dtype_changed: list[dict[str, str]] = []
    null_deltas: list[dict[str, Any]] = []

    for name in sorted(before_cols & after_cols):
        if name == row_id_col:
            continue
        before = utils.first_column_as_series(df_before, name)
        after = utils.first_column_as_series(df_after, name)
        if not is_dtype_equal(before.dtype, after.dtype) and not _same_values(
            before, after
        ):
            dtype_changed.append({
                "name": name,
                "before_dtype": str(before.dtype),
                "after_dtype": str(after.dtype),
            })
        delta = int(after.isna().sum()) - int(before.isna().sum())
        if delta:
            null_deltas.append({
                "column": name,
                "missing_before": int(before.isna().sum()),
                "missing_after": int(after.isna().sum()),
                "delta": delta,
            })

    null_deltas.sort(key=lambda row: abs(row["delta"]), reverse=True)
    null_deltas = null_deltas[:null_top_k]

    row_stats = utils.summarize_cleaning_row_effects(
        df_before, df_after, row_id_col=row_id_col
    )
    rows: dict[str, Any] = {
        "n_before": row_stats["n_in"],
        "n_after": row_stats["n_out"],
        "rows_removed_by_id": row_stats["rows_removed_by_id"],
        "rows_added_by_id": row_stats["rows_added_by_id"],
    }

    return {
        "rows": rows,
        "columns": {
            "dropped": dropped,
            "added": added,
            "dtype_changed": dtype_changed,
        },
        "null_deltas": null_deltas,
        "value_changes": _value_changes_by_row_id(
            df_before,
            df_after,
            row_id_col=row_id_col,
        ),
    }


def outcome_facts_show_any_change(facts: dict[str, Any]) -> bool:
    """True when before/after differ in shape, columns, dtypes, or null counts."""
    rows = facts["rows"]
    if rows["n_before"] != rows["n_after"]:
        return True
    if rows.get("rows_removed_by_id") or rows.get("rows_added_by_id"):
        return True
    cols = facts["columns"]
    if cols["dropped"] or cols["added"] or cols["dtype_changed"]:
        return True
    if facts["null_deltas"]:
        return True
    return bool(facts.get("value_changes"))


def format_outcome_summary_markdown(
    facts: dict[str, Any],
    *,
    row_id_col: str | None = None,
) -> str:
    """Return markdown for Streamlit."""
    lines: list[str] = []
    rows = facts["rows"]
    lines.append("**Rows**")
    lines.append(f"- Row count: **{rows['n_before']:,}** → **{rows['n_after']:,}**")
    if rows.get("rows_removed_by_id") is not None:
        lines.append(
            f"- Row ids removed (upload only): **{rows['rows_removed_by_id']:,}**"
        )
        lines.append(
            f"- Row ids added (cleaned only): **{rows['rows_added_by_id']:,}**"
        )

    cols = facts["columns"]
    lines.append("")
    lines.append("**Columns**")
    dropped = cols["dropped"]
    added = cols["added"]
    lines.append(
        f"- Dropped ({len(dropped)}): "
        + (
            ", ".join(f"`{_column_heading(c, row_id_col=row_id_col)}`" for c in dropped)
            if dropped
            else "—"
        )
    )
    lines.append(
        f"- Added ({len(added)}): "
        + (
            ", ".join(f"`{_column_heading(c, row_id_col=row_id_col)}`" for c in added)
            if added
            else "—"
        )
    )

    dtype_changed = cols["dtype_changed"]
    if dtype_changed:
        lines.append("")
        lines.append("**Dtype Changes**")
        for entry in dtype_changed:
            lines.append(
                f"- `{_column_heading(entry['name'], row_id_col=row_id_col)}`: "
                f"`{entry['before_dtype']}` → `{entry['after_dtype']}`"
            )

    if facts.get("value_changes"):
        lines.append("")
        lines.append("**Changed Values**")
        for row in facts["value_changes"]:
            lines.append(
                f"- `{_column_heading(row['column'], row_id_col=row_id_col)}`: "
                f"{row['changed_values']:,} changed"
            )

    if facts["null_deltas"]:
        lines.append("")
        lines.append("**Missing Value Count Changes (Top by |Δ|)**")
        for row in facts["null_deltas"]:
            lines.append(
                f"- `{_column_heading(row['column'], row_id_col=row_id_col)}`: "
                f"{row['missing_before']} → "
                f"{row['missing_after']} (Δ {row['delta']:+d})"
            )

    return "\n".join(lines)
