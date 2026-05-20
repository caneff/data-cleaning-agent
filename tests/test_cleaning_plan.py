import pytest

from data_cleaning_agent.cleaning_plan import (
    CleaningPlan,
    default_plan_from_summary,
    format_plan_summary_markdown,
    plan_display_json,
)


@pytest.mark.unit
def test_default_plan_seeds_coerce_columns_from_summary(mixed_df, summary) -> None:
    plan = default_plan_from_summary(summary, row_id_col="__agent_row_id__")
    assert "signup_date" in plan.coerce_datetime_columns
    assert "income_str" in plan.coerce_numeric_columns
    assert "is_active" in plan.coerce_bool_columns


@pytest.mark.unit
def test_default_plan_has_no_protected_columns_by_default(mixed_df, summary) -> None:
    plan = default_plan_from_summary(summary, row_id_col="__agent_row_id__")
    assert plan.protected_columns == []


@pytest.mark.unit
def test_format_plan_summary_markdown_is_readable() -> None:
    plan = CleaningPlan(
        skip_steps=["impute"],
        protected_columns=["country"],
        drop_high_missing_threshold=0.3,
        coerce_datetime_columns=("signup_date",),
        coerce_numeric_columns=("income_str",),
        impute_numeric_columns=("age",),
    )
    text = format_plan_summary_markdown(plan)
    assert "__agent_row_id__" not in text
    assert "`country`" in text
    assert "~~impute~~" in text
    assert "30%" in text or "0.3" in text
    assert "`signup_date`" in text
    assert "#### Pipeline steps" in text
    assert "#### Impute" in text
    assert "synthetic row id" not in text


@pytest.mark.unit
def test_format_plan_summary_omits_protected_section_without_protected_columns(
    mixed_df,
    summary,
) -> None:
    plan = default_plan_from_summary(summary, row_id_col="__agent_row_id__")
    text = format_plan_summary_markdown(plan)
    assert "#### Protected columns" not in text


@pytest.mark.unit
def test_plan_display_json_includes_user_protected_columns() -> None:
    plan = CleaningPlan(
        protected_columns=["country"],
        coerce_numeric_columns=("income_str",),
    )
    text = plan_display_json(plan)
    assert "__agent_row_id__" not in text
    assert "country" in text
    assert "income_str" in text


@pytest.mark.unit
def test_cleaning_plan_rejects_unknown_skip_step() -> None:
    with pytest.raises(ValueError, match="unknown skip_steps"):
        CleaningPlan(skip_steps=["not_a_step"])
