from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from analysis import execute_operation
from llm import ask_llm, ask_llm_json


OperationName = Literal[
    "average",
    "sum",
    "minimum",
    "maximum",
    "median",
    "unique_count",
    "count",
    "group_average",
    "group_sum",
    "group_count",
    "top_group",          # ← make sure this exists
    "top_category",
    "correlation",
    "value_count",
    "missing_values",
    "duplicate_rows",
    "dataset_summary",
]
ComparisonOperator = Literal["eq", "gt", "gte", "lt", "lte"]


class QueryPlan(BaseModel):
    operation: OperationName
    column: Optional[str] = None
    group_by: Optional[str] = None
    second_column: Optional[str] = None
    date_column: Optional[str] = None
    value: Optional[Any] = None
    operator: Optional[ComparisonOperator] = None
    frequency: Optional[Literal["D", "W", "M", "Q", "Y"]] = None
    limit: int = Field(default=10, ge=1, le=20)

    @field_validator("value")
    @classmethod
    def clean_value(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


def _schema_for_prompt(df) -> list[dict[str, Any]]:
    schema = []
    for col in df.columns:
        s = df[col]
        schema.append({
            "name": str(col),
            "dtype": str(s.dtype),
            "unique_values": int(s.nunique(dropna=True)),
            "samples": [str(v) for v in s.dropna().head(3).tolist()],
        })
    return schema


def classify_question(question: str, df) -> QueryPlan:
    schema = _schema_for_prompt(df)
    prompt = f"""
You are the analytical planning layer of a professional business intelligence application.

Dataset schema:
{schema}

User request:
{question}

Select exactly one operation from:
average, sum, minimum, maximum, median, std, count,
group_average, group_sum, group_count, top_category, value_count,
filter_count, filter_average, filter_sum, correlation, missing_values,
duplicate_rows, dataset_summary, period_change, time_trend.

Rules:
- Use ONLY columns present in the schema.
- Infer column intent from names and samples; never invent a column.
- "revenue", "sales", "price", "salary", "amount", etc. must map to the most plausible real numeric column.
- "which category has the highest average X" -> group_average.
- "which group has the most records" -> group_count.
- "how many records meet condition" -> filter_count.
- "what changed over time" -> period_change or time_trend when a date field exists.
- For trends, choose frequency M unless the user explicitly requests another granularity.
- For descriptive requests about the dataset -> dataset_summary.
- Never return executable code.

Return the structured query plan only.
"""
    plan = ask_llm_json(prompt, QueryPlan)
    valid = set(df.columns)
    for requested in (plan.column, plan.group_by, plan.second_column, plan.date_column):
        if requested is not None and requested not in valid:
            raise ValueError(f"AI selected an unknown column: {requested}")
    return plan


def answer_question(df, question: str) -> tuple[str, QueryPlan, dict[str, Any]]:
    plan = classify_question(question, df)
    exact = execute_operation(df, plan.model_dump())
    prompt = f"""
You are a senior analytics consultant.

User request:
{question}

Verified analytical plan:
{plan.model_dump()}

Verified result calculated by Python:
{exact}

Write a concise business-quality answer.
Separate facts from interpretation.
Do not invent numbers, causes, or business context.
For correlation, state that correlation does not establish causation.
For period change, state both periods and the percentage change when available.
"""
    explanation = ask_llm(
        prompt,
        max_output_tokens=500,
        system_instruction="Explain only verified analytical results. Accuracy takes priority over style.",
    )
    return explanation, plan, exact
