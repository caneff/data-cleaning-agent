"""Source Row Identity policy for cleaning flows."""

from __future__ import annotations

import dataclasses
from typing import Hashable

import pandas as pd

import data_cleaning_agent.cleaners as cleaners

DEFAULT_SOURCE_ROW_IDENTITY_LABEL = "__agent_row_id__"


@dataclasses.dataclass(frozen=True)
class SourceRowIdentityPolicy:
    """Policy for carrying Source Row Identity through a cleaning flow."""

    identity_label: Hashable
    protected_source_columns: tuple[Hashable, ...] = ()

    def protected_column_labels(self, *, names_normalized: bool) -> tuple[Hashable, ...]:
        """Return Protected Column labels for the current naming stage."""
        if not names_normalized:
            return self.protected_source_columns
        return tuple(
            cleaners.normalize_column_label(column)
            for column in self.protected_source_columns
        )

    def cleaning_exclusion_labels(
        self, *, names_normalized: bool
    ) -> tuple[Hashable, ...]:
        """Return labels that destructive cleaning steps must not change."""
        return (
            self.identity_label,
            *self.protected_column_labels(names_normalized=names_normalized),
        )


@dataclasses.dataclass(frozen=True)
class PreparedSourceFrame:
    """Source data with Source Row Identity attached."""

    frame: pd.DataFrame
    policy: SourceRowIdentityPolicy


@dataclasses.dataclass(frozen=True)
class CleaningRun:
    """One cleaning execution carrying Source Row Identity through the flow."""

    source_frame: pd.DataFrame
    policy: SourceRowIdentityPolicy
    cleaned_frame: pd.DataFrame | None = None

    def with_cleaned_frame(self, cleaned: pd.DataFrame) -> CleaningRun:
        """Return this run with cleaned data attached."""
        return dataclasses.replace(self, cleaned_frame=cleaned)

    def source_user_data(self) -> pd.DataFrame:
        """Return source data without internal Source Row Identity."""
        return export_cleaned_frame(self.source_frame, self.policy)

    def cleaned_user_data(self) -> pd.DataFrame | None:
        """Return cleaned data without internal Source Row Identity."""
        if self.cleaned_frame is None:
            return None
        return export_cleaned_frame(self.cleaned_frame, self.policy)


def _available_identity_label(columns: pd.Index) -> str:
    label = DEFAULT_SOURCE_ROW_IDENTITY_LABEL
    if label not in columns:
        return label
    n = 1
    while f"{label}_{n}" in columns:
        n += 1
    return f"{label}_{n}"


def prepare_source_frame(
    source: pd.DataFrame,
    *,
    protected_columns: tuple[Hashable, ...] = (),
) -> PreparedSourceFrame:
    """Return source data with Source Row Identity assigned."""
    out = source.copy()
    identity_label = _available_identity_label(source.columns)
    out.insert(
        0,
        identity_label,
        pd.Series(
            [str(i) for i in range(len(source))],
            index=source.index,
            dtype="string",
        ),
    )
    policy = SourceRowIdentityPolicy(
        identity_label=identity_label,
        protected_source_columns=protected_columns,
    )
    return PreparedSourceFrame(frame=out, policy=policy)


def start_cleaning_run(
    source: pd.DataFrame,
    *,
    protected_columns: tuple[Hashable, ...] = (),
) -> CleaningRun:
    """Start a Cleaning Run by attaching Source Row Identity to source data."""
    prepared = prepare_source_frame(source, protected_columns=protected_columns)
    return CleaningRun(source_frame=prepared.frame, policy=prepared.policy)


def export_cleaned_frame(
    cleaned: pd.DataFrame,
    policy: SourceRowIdentityPolicy,
) -> pd.DataFrame:
    """Return cleaned data without internal Source Row Identity."""
    return cleaned.drop(columns=[policy.identity_label], errors="ignore").copy()
