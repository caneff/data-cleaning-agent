import pandas as pd
import pytest

import data_cleaning_agent.source_row_identity as source_row_identity


@pytest.mark.unit
def test_assigns_source_row_identity_to_source_rows() -> None:
    source = pd.DataFrame({"name": ["Ada", "Grace"]})

    prepared = source_row_identity.prepare_source_frame(source)

    assert prepared.policy.identity_label in prepared.frame.columns
    assert prepared.frame[prepared.policy.identity_label].tolist() == ["0", "1"]
    assert source_row_identity.DEFAULT_SOURCE_ROW_IDENTITY_LABEL not in source.columns


@pytest.mark.unit
def test_source_row_identity_avoids_user_column_collision() -> None:
    default_label = source_row_identity.DEFAULT_SOURCE_ROW_IDENTITY_LABEL
    source = pd.DataFrame({default_label: ["user-1", "user-2"], "name": ["Ada", "Grace"]})

    prepared = source_row_identity.prepare_source_frame(source)

    assert prepared.policy.identity_label != default_label
    assert prepared.policy.identity_label in prepared.frame.columns
    assert prepared.frame[prepared.policy.identity_label].tolist() == ["0", "1"]
    assert prepared.frame[default_label].tolist() == ["user-1", "user-2"]


@pytest.mark.unit
def test_protected_column_labels_follow_standard_normalization() -> None:
    source = pd.DataFrame({"Country Name": ["US"], "score": [10]})

    prepared = source_row_identity.prepare_source_frame(
        source,
        protected_columns=("Country Name",),
    )

    assert prepared.policy.protected_column_labels(names_normalized=False) == (
        "Country Name",
    )
    assert prepared.policy.protected_column_labels(names_normalized=True) == (
        "country_name",
    )


@pytest.mark.unit
def test_cleaning_exclusion_labels_include_identity_and_protected_columns() -> None:
    source = pd.DataFrame({"Country Name": ["US"], "score": [10]})

    prepared = source_row_identity.prepare_source_frame(
        source,
        protected_columns=("Country Name",),
    )

    assert prepared.policy.cleaning_exclusion_labels(names_normalized=True) == (
        prepared.policy.identity_label,
        "country_name",
    )


@pytest.mark.unit
def test_export_cleaned_frame_removes_identity_and_preserves_user_collision() -> None:
    default_label = source_row_identity.DEFAULT_SOURCE_ROW_IDENTITY_LABEL
    source = pd.DataFrame({default_label: ["user-1"], "name": ["Ada"]})
    prepared = source_row_identity.prepare_source_frame(source)

    exported = source_row_identity.export_cleaned_frame(
        prepared.frame,
        prepared.policy,
    )

    assert prepared.policy.identity_label not in exported.columns
    assert exported[default_label].tolist() == ["user-1"]
    assert exported["name"].tolist() == ["Ada"]
