import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
from detective import performance as pf, timeseries as tsm  # noqa: E402
from detective.copilot import Copilot, Intent, detect_intent  # noqa: E402
from detective.dataset import Dataset, load_data  # noqa: E402
from detective.engine import analyze  # noqa: E402
from detective import llm  # noqa: E402


def build(name):
    ds = Dataset(load_data(str(ROOT / "sample_data" / f"{name}.csv"), name), name)
    A = analyze(ds)
    return ds, A, Copilot(ds, A)


@pytest.fixture(scope="module")
def retail():
    return build("retail_sales")


# ---------------- schema inference is domain-agnostic ----------------
def test_roles_retail(retail):
    ds, _, _ = retail
    assert ds.primary == "Revenue" and ds.time_spec["col"] == "OrderDate"
    assert ds.cols["OrderID"].role == "id" and ds.cols["Customer"].role == "entity"
    assert ds.kind("Revenue") == "additive" and ds.kind("UnitPrice") == "intensity"


def test_roles_housing_uses_event_year_not_attribute_year():
    ds, _, _ = build("house_prices")
    assert ds.time_spec["col"] == "YrSold" and ds.primary == "SalePrice"


def test_outcome_flag_becomes_primary_for_hr():
    ds, _, _ = build("employee_attrition")
    assert ds.primary == "Attrition" and ds.kind("Attrition") == "rate" and not ds.has_time


# ---------------- analytics correctness ----------------
def test_partial_period_excluded(retail):
    _, A, _ = retail
    assert A.ts.partial_last is not None and A.ts.partial_last["coverage"] < 0.2
    assert A.pop["latest_label"] == "Sep 2026"


def test_decomposition_is_exact_additive_and_mean(retail):
    ds, A, _ = retail
    for d in A.change["dimensions"]:
        assert d["table"]["delta"].sum() == pytest.approx(A.change["pop"]["change"], rel=1e-9)
    ts = tsm.build_ts(ds, "UnitPrice")
    ex = pf.explain_change(ds, "UnitPrice", ts)
    for d in ex["dimensions"]:
        assert d["table"]["delta"].sum() == pytest.approx(ex["pop"]["change"], abs=1e-9)


def test_planted_event_is_found(retail):
    _, A, _ = retail
    top = A.change["dimensions"][0]
    assert top["dim"] == "Region" and top["lead"]["group"] == "West"
    assert A.change["volume_value"]["total"] == pytest.approx(A.pop["change"], rel=1e-9)


def test_planted_anomalies_and_duplicates_found(retail):
    _, A, _ = retail
    assert A.anomalies["table"].iloc[0]["Revenue"] == 240000.0
    assert A.quality["duplicate_rows"] >= 25 and A.quality["id_duplicates"] >= 25


def test_forecast_is_backtested_and_labelled(retail):
    _, A, _ = retail
    assert A.forecast["available"] and A.forecast["reliability"] in {"High", "Medium", "Low"}
    assert A.forecast["backtest_mape"] is not None


def test_noise_not_reported_as_driver_or_change():
    ds, A, _ = build("employee_attrition")
    labels = [d["label"] for d in A.drivers["drivers"]]
    assert "Over Time" in labels and "Years At Company" not in labels
    ds2, A2, _ = build("house_prices")
    assert not any(f.key == "change_driver" for f in A2.findings)  # +5.6% on random noise is not a "story"


def test_query_semantics_distinguish_volume_revenue_and_frequency(retail):
    ds, _, cp = retail

    # "sold" means sales volume/units, not revenue and not row frequency.
    sold_plan = cp._plan("Which product sold the most?")
    assert sold_plan.intent == "top_performers"
    assert sold_plan.dimension == "ProductCategory"
    assert sold_plan.measure == "Quantity"

    # "generated revenue" explicitly selects the monetary measure.
    revenue_plan = cp._plan("Which product generated the most revenue?")
    assert revenue_plan.intent == "top_performers"
    assert revenue_plan.dimension == "ProductCategory"
    assert revenue_plan.measure == "Revenue"

    # Frequency wording intentionally remains count-based.
    freq_plan = cp._plan("Which product appears most frequently?")
    assert freq_plan.intent == "top_performers"
    assert freq_plan.dimension == "ProductCategory"
    assert freq_plan.measure is None or freq_plan.measure == "Quantity"
    assert freq_plan.aggregation == "count"


def test_sold_volume_measure_is_additive_for_shipped_fields():
    df = pd.DataFrame({
        "Product": ["A", "B", "A", "C", "B", "A"],
        "Boxes Shipped": [10, 20, 15, 5, 25, 12],
        "Revenue": [100, 300, 200, 50, 450, 120],
    })
    ds = Dataset(df, "semantic-test")
    A = analyze(ds)
    cp = Copilot(ds, A)
    plan = cp._plan("Which product sold the most?")
    assert plan.measure == "Boxes Shipped"
    assert ds.kind("Boxes Shipped") == "additive"
    answer = cp.ask("Which product sold the most?")
    assert "B" in answer.headline
    assert "45" in " ".join(answer.facts)


def test_sold_most_returns_quantity_not_revenue(retail):
    ds, _, cp = retail
    a = cp.ask("Which product sold the most?")
    qty = ds.df.groupby("ProductCategory")["Quantity"].sum().sort_values(ascending=False)
    top = qty.index[0]
    assert top in a.headline
    assert f"{int(qty.iloc[0]):,}" in " ".join(a.facts)


# ---------------- copilot behaviour ----------------
def test_why_question_runs_multistep_analysis(retail):
    _, _, cp = retail
    a = cp.ask("Why did revenue decline?")
    text = " ".join(a.facts)
    assert "Volume vs value" in text and "West" in text and a.recommendations and a.interpretation
    assert a.figures and not a.limitations


def test_wrong_premise_is_called_out(retail):
    _, _, cp = retail
    assert "Premise check" in cp.ask("Why did revenue increase?").facts[0]


def test_missing_time_gives_exact_limitation():
    _, _, cp = build("employee_attrition")
    a = cp.ask("Show the trend")
    assert "no suitable time/date column was detected" in a.limitations[0]


def test_missing_concept_is_not_guessed():
    _, _, cp = build("house_prices")
    a = cp.ask("What is the total profit?")
    assert a.limitations and "profit" in a.limitations[0] and not a.facts


def test_out_of_scope_declined(retail):
    _, _, cp = retail
    a = cp.ask("Why is the sky blue?")
    assert a.limitations and not a.facts


def test_simple_aggregates_and_filters():
    ds, _, cp = build("house_prices")
    a = cp.ask("What is the average house price?")
    assert f"{ds.df.SalePrice.mean():,.0f}" in a.headline
    b = cp.ask("What is the average price in Downtown?")
    assert "Downtown" in b.basis and f"{ds.df[ds.df.Neighborhood == 'Downtown'].SalePrice.mean():,.0f}" in b.headline
    c = cp.ask("How many houses have price above 300000?")
    assert str(int((ds.df.SalePrice > 300000).sum())) in c.headline


def test_every_documented_question_is_answerable(retail):
    _, _, cp = retail
    qs = ["Why did revenue decline?", "Which region is underperforming?", "Which product contributes the most revenue?",
          "What changed compared with the previous period?", "What are the strongest potential drivers?",
          "Which customers are unusually valuable?", "Which records look suspicious?",
          "What are the most important findings?", "What should management investigate?", "Create an executive summary."]
    for q in qs:
        a = cp.ask(q)
        assert (a.facts or a.limitations) and a.headline, q


def test_answers_never_expose_internals(retail):
    _, _, cp = retail
    bad = ["{'", "intent", "operation=", "QueryPlan", "system instruction", "Traceback", "pydantic"]
    for q in ["Why did revenue decline?", "Create an executive summary.", "gibberish xyz", "Which records look suspicious?"]:
        a = cp.ask(q)
        blob = a.evidence_text() + " ".join(a.limitations) + (a.interpretation or "") + a.basis
        assert not any(b in blob for b in bad), (q, blob[:200])


def test_works_without_any_llm(monkeypatch, retail):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AI_PROVIDER", "none")
    assert not llm.available()
    _, _, cp = retail
    assert cp.ask("Why did revenue decline?").facts


# ---------------- LLM safety layer ----------------
def test_numeric_grounding_rejects_invented_numbers():
    ev = "Revenue fell 38.9% to 233,760 in Sep 2026 from 382,824."
    assert llm.grounded("Revenue dropped about 38.9%, to 233,760.", ev)
    assert not llm.grounded("The fall is 38.9%; 3 regions moved.", ev)
    assert llm.grounded("Revenue fell 38.9% in 2026.", ev)
    assert not llm.grounded("Revenue fell 41% last month.", ev)
    assert not llm.grounded("Losses reached 1.5M.", ev)


# ---------------- robustness on messy, heterogeneous input ----------------
def test_messy_strings_and_excel_roundtrip():
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({
        "Txn Date": pd.date_range("2025-01-01", periods=n, freq="3D").strftime("%d/%m/%Y"),
        "Amount ($)": [f"${x:,.2f}" for x in rng.uniform(10, 900, n)],
        "Channel": rng.choice(["web", "store", "app"], n),
        "Rating": rng.integers(1, 6, n), "amount ($)": rng.normal(5, 1, n),
    })
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    ds = Dataset(load_data(buf, "x.xlsx"), "x")
    assert pd.api.types.is_numeric_dtype(ds.df["Amount ($)"]) and ds.has_time
    A = analyze(ds)
    assert A.findings and Copilot(ds, A).ask("What is the average amount?").facts


def test_tiny_and_text_only_datasets_do_not_crash():
    tiny = Dataset(load_data(io.BytesIO(b"a,b\n1,x\n2,y\n3,x\n"), "t.csv"), "t")
    A = analyze(tiny)
    assert A.quality["score"] > 0
    assert Copilot(tiny, A).ask("Create an executive summary.").headline
    text_only = Dataset(pd.DataFrame({"city": ["A", "B", "A", "C"] * 10, "tier": ["x", "y"] * 20}), "t")
    A2 = analyze(text_only)
    assert Copilot(text_only, A2).ask("What are the key findings?").headline
    assert Copilot(text_only, A2).ask("What is the average city?").headline


def test_intents_are_whitelisted():
    assert detect_intent("why did sales fall") == "explain_change"
    assert detect_intent("zzz") is None
    assert "explain_change" in Intent.__args__
