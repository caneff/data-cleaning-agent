"""Tests for LangGraph wiring and LightweightDataCleaningAgent plan path."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import data_cleaning_agent.source_row_identity as source_row_identity
from data_cleaning_agent.cleaning_plan import CleaningPlan, default_plan_from_summary
from data_cleaning_agent.data_cleaning_agent import (
    LightweightDataCleaningAgent,
    make_lightweight_data_cleaning_agent,
)
from data_cleaning_agent.plan_generation import FIX_PLAN_PROMPT_TEMPLATE

_ROW_ID = source_row_identity.DEFAULT_SOURCE_ROW_IDENTITY_LABEL


def _df_with_row_id(mixed_df: pd.DataFrame) -> pd.DataFrame:
    out = mixed_df.copy()
    out.insert(0, _ROW_ID, [str(i) for i in range(len(out))])
    return out


def _valid_plan_payload(summary) -> dict:
    example = default_plan_from_summary(summary, row_id_col=_ROW_ID)
    return {
        **asdict(example),
        "protected_columns": ["country"],
    }


@pytest.mark.unit
def test_fix_plan_prompt_formats_with_expected_placeholders() -> None:
    rendered = FIX_PLAN_PROMPT_TEMPLATE.format(
        user_instructions="protect country",
        all_datasets_summary="Rows: 5",
        pipeline_step_ids="normalize_names, impute",
        plan_snippet='{"skip_steps": []}',
        error="ValueError: unknown skip_steps",
        row_id_col=_ROW_ID,
    )
    assert "CleaningPlan" in rendered
    assert "```json" in rendered
    assert "unknown skip_steps" in rendered
    assert _ROW_ID not in rendered


@pytest.mark.unit
def test_invoke_agent_runs_pipeline_with_mock_llm(
    mixed_df, summary, monkeypatch
) -> None:
    payload = _valid_plan_payload(summary)
    fake_model = RunnableLambda(
        lambda _prompt: AIMessage(content=json.dumps(payload)),
    )
    monkeypatch.setattr(
        "data_cleaning_agent.data_cleaning_agent.plan_generation.generate_cleaning_plan",
        lambda model, source_df, user_instructions=None, **kwargs: CleaningPlan(
            **payload
        ),
    )

    agent = LightweightDataCleaningAgent(model=fake_model)
    df = _df_with_row_id(mixed_df)
    agent.invoke_agent(df, user_instructions="protect country", max_retries=0)

    cleaned = agent.get_data_cleaned()
    assert cleaned is not None
    assert len(cleaned) == len(df)
    assert agent.response is not None
    assert agent.response.get("data_cleaner_error") is None
    assert agent.response.get("cleaning_plan") is not None


@pytest.mark.unit
def test_invoke_agent_keeps_identity_internal_for_cleaning_run(
    mixed_df, summary, monkeypatch
) -> None:
    payload = _valid_plan_payload(summary)
    fake_model = RunnableLambda(
        lambda _prompt: AIMessage(content=json.dumps(payload)),
    )
    monkeypatch.setattr(
        "data_cleaning_agent.data_cleaning_agent.plan_generation.generate_cleaning_plan",
        lambda model, source_df, user_instructions=None, **kwargs: CleaningPlan(
            **payload
        ),
    )

    agent = LightweightDataCleaningAgent(model=fake_model)
    agent.invoke_agent(mixed_df, user_instructions="protect country", max_retries=0)

    run = agent.get_cleaning_run()
    assert run is not None
    assert run.policy.identity_label in run.source_frame.columns
    assert run.cleaned_frame is not None
    assert run.policy.identity_label in run.cleaned_frame.columns
    input_df = agent.get_input_dataframe()
    cleaned = agent.get_data_cleaned()
    assert input_df is not None
    assert cleaned is not None
    assert run.policy.identity_label not in input_df.columns
    assert run.policy.identity_label not in cleaned.columns


@pytest.mark.unit
def test_invoke_agent_generates_plan_with_active_source_row_identity_label(
    mixed_df, summary, monkeypatch
) -> None:
    payload = _valid_plan_payload(summary)
    seen: dict[str, Any] = {}

    def fake_generate(model, source_df, user_instructions=None, **kwargs):
        seen["row_id_col"] = kwargs["row_id_col"]
        seen["source_df"] = source_df
        return CleaningPlan(**payload)

    monkeypatch.setattr(
        "data_cleaning_agent.data_cleaning_agent.plan_generation.generate_cleaning_plan",
        fake_generate,
    )

    agent = LightweightDataCleaningAgent(model=object())
    agent.invoke_agent(_df_with_row_id(mixed_df), max_retries=0)

    run = agent.get_cleaning_run()
    assert run is not None
    assert seen["row_id_col"] == run.policy.identity_label
    assert run.policy.identity_label != _ROW_ID
    assert run.policy.identity_label in seen["source_df"].columns


@pytest.mark.unit
def test_generate_and_execute_stored_cleaning_plan(
    mixed_df, summary, monkeypatch
) -> None:
    payload = _valid_plan_payload(summary)
    fake_model = RunnableLambda(
        lambda _prompt: AIMessage(content=json.dumps(payload)),
    )
    monkeypatch.setattr(
        "data_cleaning_agent.data_cleaning_agent.plan_generation.generate_cleaning_plan",
        lambda model, source_df, user_instructions=None, **kwargs: CleaningPlan(
            **payload
        ),
    )

    agent = LightweightDataCleaningAgent(model=fake_model)
    agent.generate_cleaning_plan(mixed_df, user_instructions="protect country")

    plan = agent.get_cleaning_plan()
    assert plan is not None
    assert "country" in plan.protected_columns

    out = agent.execute_stored_cleaning()
    assert out.get("data_cleaner_error") is None
    assert agent.get_data_cleaned() is not None


@pytest.mark.unit
def test_generate_cleaning_plan_uses_active_source_row_identity_label(
    mixed_df, summary, monkeypatch
) -> None:
    payload = _valid_plan_payload(summary)
    seen: dict[str, Any] = {}

    def fake_generate(model, source_df, user_instructions=None, **kwargs):
        seen["row_id_col"] = kwargs["row_id_col"]
        seen["source_df"] = source_df
        return CleaningPlan(**payload)

    monkeypatch.setattr(
        "data_cleaning_agent.data_cleaning_agent.plan_generation.generate_cleaning_plan",
        fake_generate,
    )

    agent = LightweightDataCleaningAgent(model=object())
    agent.generate_cleaning_plan(_df_with_row_id(mixed_df))

    run = agent.get_cleaning_run()
    assert run is not None
    assert seen["row_id_col"] == run.policy.identity_label
    assert run.policy.identity_label != _ROW_ID
    assert run.policy.identity_label in seen["source_df"].columns


@pytest.mark.unit
def test_stored_cleaning_uses_captured_cleaning_run_without_public_identity(
    mixed_df, summary, monkeypatch
) -> None:
    payload = _valid_plan_payload(summary)
    fake_model = RunnableLambda(
        lambda _prompt: AIMessage(content=json.dumps(payload)),
    )
    monkeypatch.setattr(
        "data_cleaning_agent.data_cleaning_agent.plan_generation.generate_cleaning_plan",
        lambda model, source_df, user_instructions=None, **kwargs: CleaningPlan(
            **payload
        ),
    )

    agent = LightweightDataCleaningAgent(model=fake_model)
    agent.generate_cleaning_plan(mixed_df, user_instructions="protect country")

    out = agent.execute_stored_cleaning()
    cleaned = pd.DataFrame(out["data_cleaned"])
    input_df = agent.get_input_dataframe()
    agent_cleaned = agent.get_data_cleaned()

    assert out.get("data_cleaner_error") is None
    assert input_df is not None
    assert agent_cleaned is not None
    assert _ROW_ID not in cleaned.columns
    assert _ROW_ID not in input_df.columns
    assert _ROW_ID not in agent_cleaned.columns


@pytest.mark.unit
def test_graph_retries_fix_on_pipeline_error(mixed_df, summary, monkeypatch) -> None:
    payload = _valid_plan_payload(summary)
    calls: list[str] = []

    def fake_generate(model, source_df, user_instructions=None, **kwargs):
        calls.append("generate")
        return CleaningPlan(**payload)

    def fake_repair(model, source_df, *, broken_plan, error, **kwargs):
        calls.append("repair")
        return CleaningPlan(**payload)

    execute_count = {"n": 0}

    def fake_run_cleaning_pipeline(
        df,
        plan,
        *,
        row_id_col=source_row_identity.DEFAULT_SOURCE_ROW_IDENTITY_LABEL,
    ):
        execute_count["n"] += 1
        if execute_count["n"] == 1:
            msg = "simulated pipeline failure"
            raise RuntimeError(msg)
        from data_cleaning_agent.cleaning_pipeline import PipelineTrace

        return df.copy(), PipelineTrace()

    monkeypatch.setattr(
        "data_cleaning_agent.data_cleaning_agent.plan_generation.generate_cleaning_plan",
        fake_generate,
    )
    monkeypatch.setattr(
        "data_cleaning_agent.data_cleaning_agent.plan_generation.fix_cleaning_plan",
        fake_repair,
    )
    monkeypatch.setattr(
        "data_cleaning_agent.data_cleaning_agent.cleaning_pipeline.run_cleaning_pipeline",
        fake_run_cleaning_pipeline,
    )

    graph = make_lightweight_data_cleaning_agent(model=object())
    df = _df_with_row_id(mixed_df)
    result = graph.invoke({
        "user_instructions": "protect country",
        "source_df": df.to_dict(),
        "max_retries": 1,
        "retry_count": 0,
    })

    assert calls == ["generate", "repair"]
    assert result.get("data_cleaner_error") is None
    assert execute_count["n"] == 2
