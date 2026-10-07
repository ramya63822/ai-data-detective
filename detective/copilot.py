"""AI analyst: natural-language questions -> verified, multi-step analysis -> structured answer.

Pipeline for every question:
  1. Plan      — map the question to a whitelisted analysis (rules; optionally refined by an LLM and validated)
  2. Execute   — deterministic pandas/statistics operations compute all evidence
  3. Compose   — facts (verified) / interpretation / recommendations (rule-based) / limitations
  4. Narrate   — an optional LLM interpretation, kept only if every number in it is present in the evidence
Nothing about plans, prompts or validation is exposed to the user.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from . import llm
from . import performance as pf
from . import timeseries as tsm
from . import visuals as vz
from .dataset import Dataset
from .engine import Analysis, analyze, headline_value
from .fmt import extract_numbers, join_and, num, pct, pretty, tokens, unit_word, value

Intent = Literal[
    "executive_summary", "key_findings", "investigate", "risks", "opportunities", "explain_change", "period_comparison",
    "trend", "forecast", "seasonality", "top_performers", "bottom_performers", "contribution", "drivers", "correlation",
    "anomalies", "valuable_entities", "segmentation", "data_quality", "aggregate", "distribution", "overview", "unsupported",
]


class Filter(BaseModel):
    column: str
    operator: Literal["eq", "gt", "gte", "lt", "lte"] = "eq"
    value: Union[str, float]


class Plan(BaseModel):
    intent: Intent
    measure: Optional[str] = None
    dimension: Optional[str] = None
    second_measure: Optional[str] = None
    aggregation: Optional[Literal["average", "sum", "median", "minimum", "maximum", "count"]] = None
    filters: list[Filter] = Field(default_factory=list)
    top_n: int = Field(default=10, ge=1, le=20)


@dataclass
class Answer:
    headline: str
    facts: list[str] = field(default_factory=list)
    interpretation: Optional[str] = None
    recommendations: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    tables: list[tuple[str, pd.DataFrame]] = field(default_factory=list)
    figures: list[Any] = field(default_factory=list)
    basis: str = ""
    suggestions: list[str] = field(default_factory=list)

    def evidence_text(self) -> str:
        parts = [self.headline] + self.facts
        for title, t in self.tables:
            parts.append(title + ": " + "; ".join(" | ".join(str(v) for v in r) for r in t.head(8).itertuples(index=False)))
        return "\n".join(parts)


# --------------------------------------------------------------------------------------
# Language helpers
# --------------------------------------------------------------------------------------
_SYN = {
    "revenue": {"sales", "turnover"}, "sales": {"revenue", "turnover"}, "turnover": {"revenue", "sales"},
    "income": {"earnings"}, "earnings": {"income"}, "customer": {"client", "buyer", "shopper"},
    "employee": {"staff", "worker"}, "attrition": {"churn", "resignation"}, "churn": {"attrition"},
    "region": {"area", "territory", "zone"}, "product": {"item", "sku"}, "category": {"type"},
}
_CONCEPTS = {"revenue", "sales", "profit", "price", "cost", "salary", "income", "margin", "spend", "expense", "turnover",
             "quantity", "attrition", "churn", "revenue"}
_STOP = {"the", "a", "an", "of", "is", "are", "what", "which", "how", "by", "for", "in", "on", "to", "and", "or", "me",
         "show", "our", "my", "was", "were", "did", "do", "does", "with", "that", "this", "per", "each", "all", "from"}


def _stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _qtokens(q: str) -> set[str]:
    base = {_stem(t) for t in re.findall(r"[a-z0-9]+", q.lower()) if t not in _STOP}
    out = set(base)
    for t in base:
        out |= {_stem(s) for s in _SYN.get(t, ())}
    return out


def _score_col(qt: set[str], name: str) -> float:
    ct = {_stem(t) for t in tokens(name)}
    if not ct:
        return 0.0
    hit = len(qt & ct)
    return hit / len(ct) if hit else 0.0


def _best(qt: set[str], names: list[str], floor: float = 0.5) -> Optional[str]:
    scored = [(_score_col(qt, n), -i, n) for i, n in enumerate(names)]
    scored = [s for s in scored if s[0] >= floor]
    return max(scored)[2] if scored else None


# --------------------------------------------------------------------------------------
# Rule-based planner (always available)
# --------------------------------------------------------------------------------------
_RULES: list[tuple[str, str]] = [
    ("executive_summary", r"executive summary|exec summary|summari[sz]e (the )?(data|dataset|findings)|management summary|board summary|brief me"),
    ("valuable_entities", r"(unusually|exceptionally|most|highly) valuable|high[- ]value|\bvip\b|best (customers|clients)|top (customers|clients)|biggest (customers|clients)"),
    ("investigate", r"should (management|we|i|leadership) (investigate|look|focus|do|prioriti[sz]e)|next steps?|what to (investigate|look into)|worth investigating|action items?|prioriti[sz]e"),
    ("key_findings", r"(important|key|main|top|biggest) (findings?|insights?|takeaways?|points?|signals?)|findings|insights|highlights|what stands out"),
    ("explain_change", r"\bwhy\b|what (caused|drove|explains?)|reasons? for|root cause|behind the (decline|drop|increase|rise|fall)|how come|what happened"),
    ("period_comparison", r"what changed|changed (compared|versus|vs|from|since)|compared (with|to) (the )?(previous|prior|last)|previous (period|month|quarter|year|week)|last (month|quarter|year|week)|(month|year|quarter|week|period)[- ]over[- ](month|year|quarter|week|period)|\byoy\b|\bmom\b|\bqoq\b|versus last|vs\.? last"),
    ("forecast", r"forecast|predict(ion)? (the )?(next|future)|projection|projected|next (\d+ )?(month|quarter|year|week|day)s?|outlook|expected (to|revenue|sales)|going forward"),
    ("seasonality", r"season|peak (month|season|period|time)|time of year|cyclic|day of (the )?week|weekday"),
    ("trend", r"\btrend|over time|growth rate|trajectory|timeline|evolv|how (has|have|is|are) .* (changed|evolved|performed)"),
    ("anomalies", r"anomal|outlier|suspicious|unusual (record|transaction|order|entry)|fraud|irregular|abnormal|strange|odd (record|value)|errors?\b|look(s)? wrong|data issues?"),
    ("drivers", r"driver|drives|influenc|affect|impact|determin|factors?|what (matters|explains)|predictors?"),
    ("correlation", r"correlat|relationship between|associated with|related to|link between"),
    ("segmentation", r"segment|cluster|personas?|types? of (customer|record|user)|groups? of (customer|record|user)"),
    ("data_quality", r"missing|null|duplicate|data quality|data health|clean(liness)?|complete(ness)?|reliab"),
    ("risks", r"\brisks?\b|concerns?|problems?|threats?|red flags?|worr|weak spots?|areas? (of|requiring) (concern|attention)"),
    ("opportunities", r"opportunit|upside|potential (for|to)|room (to|for) (grow|improve)"),
    ("bottom_performers", r"underperform|under-perform|worst|lowest|bottom|weakest|poor(est)?|least|smallest|laggard|struggl|declin(ing|ed) (the )?most"),
    ("contribution", r"contribut|share of|proportion|percentage of|\bmix\b|breakdown|split|concentrat|pareto|depend(ent|ency|s)? on|accounts? for"),
    ("top_performers", r"\bbest\b|\btop\b|highest|most|largest|biggest|strongest|leading|leader|outperform|greatest|better"),
    ("distribution", r"distribution|spread|histogram|\brange\b|skew|variab|dispersion|percentile"),
    # Direct data-retrieval requests are valid dataset operations and should not be
    # forced through analytical ranking/aggregation intents.
    ("record_lookup", r"\b(?:show|give|list|display|fetch|provide|return|get)\b.*\b(?:ids?|identifiers?|records?|rows?|entries?)\b|\b(?:student|customer|employee|user|client|account)\s+(?:ids?|identifiers?|records?|rows?)\b|\b(?:ids?|identifiers?)\s+of\s+(?:students?|customers?|employees?|users?|clients?|accounts?)\b"),
    ("aggregate", r"average|mean\b|median|total|\bsum\b|minimum|maximum|\bmin\b|\bmax\b|how (many|much)|count|number of|typical|standard deviation|std"),
    ("overview", r"overview|describe|tell me about|what is in|what's in|about (this|the) (data|dataset)|columns|fields|schema"),
]
_NEEDS_TIME = {"explain_change", "period_comparison", "trend", "forecast", "seasonality"}
_OPS = [
    ("count", r"how many|count|number of"), ("median", r"median"), ("average", r"average|mean\b|typical|avg"),
    ("minimum", r"minimum|\bmin\b|lowest value|smallest"), ("maximum", r"maximum|\bmax\b|highest value|largest value"),
    ("sum", r"total|\bsum\b|how much"),
]


def detect_intent(q: str) -> Optional[str]:
    ql = q.lower()
    for intent, pat in _RULES:
        if re.search(pat, ql):
            return intent
    return None


def _parse_filters(q: str, ds: Dataset, measure: Optional[str]) -> list[Filter]:
    ql = q.lower()
    out: list[Filter] = []
    for dim in ds.categorical_dimensions + ds.entities[:0]:
        for v in ds.df[dim].dropna().unique()[:60]:
            sv = str(v).strip()
            if len(sv) >= 2 and re.search(rf"(?<![a-z0-9]){re.escape(sv.lower())}(?![a-z0-9])", ql):
                out.append(Filter(column=dim, operator="eq", value=sv))
    m = re.search(r"(>=|<=|>|<|at least|at most|above|over|greater than|more than|below|under|less than)\s*\$?(\d[\d,]*\.?\d*)", ql)
    if m and (measure or ds.primary):
        op = {">=": "gte", "at least": "gte", ">": "gt", "above": "gt", "over": "gt", "greater than": "gt", "more than": "gt",
              "<=": "lte", "at most": "lte", "<": "lt", "below": "lt", "under": "lt", "less than": "lt"}[m.group(1)]
        out.append(Filter(column=measure or ds.primary, operator=op, value=float(m.group(2).replace(",", ""))))
    return out


_SEMANTIC_MEASURE_HINTS: list[tuple[str, set[str], tuple[str, ...]]] = [
    # Volume / activity concepts: "sold", "shipped", "units", etc. should prefer an
    # additive volume field over revenue when the user asks what sold the most.
    ("volume", {
        "sold", "sell", "shipped", "ship", "boxes", "box", "units", "unit",
        "quantity", "qty", "volume", "ordered", "orders", "bookings", "shipments",
        "transactions", "transaction", "tickets",
    }, ("quantity", "qty", "units", "unit", "volume", "shipped", "boxes", "orders", "bookings", "shipments", "tickets", "items")),
    ("revenue", {"revenue", "sales", "turnover", "income", "earnings", "amount", "money", "value"},
     ("revenue", "sales", "turnover", "income", "earnings", "amount", "gmv", "value")),
    ("price", {"price", "pricing", "rate", "cost per", "unitprice", "unit", "fare", "fee"},
     ("price", "unitprice", "unit_price", "rate", "fare", "fee", "cost")),
    ("profit", {"profit", "margin", "profitability"}, ("profit", "margin")),
    ("cost", {"cost", "expense", "spend", "spending"}, ("cost", "expense", "spend")),
    ("salary", {"salary", "pay", "compensation", "wage"}, ("salary", "pay", "compensation", "wage")),
]


def _measure_from_semantics(q: str, measures: list[str]) -> Optional[str]:
    """Pick a measure from the meaning of the question, not only literal column-name overlap."""
    ql = q.lower()
    qt = _qtokens(q)

    def score(name: str, aliases: tuple[str, ...]) -> float:
        raw = name.lower().replace("_", " ").replace("-", " ")
        nt = {_stem(t) for t in re.findall(r"[a-z0-9]+", raw)}
        s = 0.0
        s += 4.0 * len(nt & {_stem(a) for a in aliases})
        s += 1.5 * len(qt & nt)
        # Prefer additive measures for volume-like questions.
        if "sales" in raw and any(w in ql for w in ("sold", "shipped", "units", "boxes", "quantity", "volume")):
            s -= 4.0
        if "revenue" in raw and any(w in ql for w in ("sold", "shipped", "units", "boxes", "quantity")):
            s -= 3.0
        return s

    for _, trigger_words, aliases in _SEMANTIC_MEASURE_HINTS:
        # Compare single-word concepts against tokenized question terms. Raw substring
        # matching caused false positives such as ``rate`` matching ``generated``.
        matched = any((w in qt) or (" " in w and w in ql) for w in trigger_words)
        if matched:
            ranked = sorted(((score(m, aliases), i, m) for i, m in enumerate(measures)), reverse=True)
            positive = [x for x in ranked if x[0] > 0]
            if positive:
                return positive[0][2]
    return None


def rule_plan(q: str, ds: Dataset) -> Plan:
    qt = _qtokens(q)
    intent = detect_intent(q)
    all_meas = ds.all_measures

    # Semantic cues override the generic "primary measure" heuristic. This is critical
    # for business wording such as "sold the most" where Revenue may be the primary KPI
    # but Quantity is the requested measure.
    semantic_measure = _measure_from_semantics(q, all_meas)
    measure = semantic_measure or _best(qt, all_meas)

    dims = ds.categorical_dimensions + ds.entities + [d for d in ds.dimensions if ds.cols[d].role != "dimension"]
    dimension = _best(qt, [d for d in dims if d != measure])
    second = None

    if intent == "correlation":
        hits = sorted(((_score_col(qt, m), m) for m in all_meas if _score_col(qt, m) >= 0.5), reverse=True)
        if len(hits) >= 2:
            measure, second = hits[0][1], hits[1][1]

    aggregation = next((op for op, pat in _OPS if re.search(pat, q.lower())), None)
    if re.search(r"frequent|frequency|appears most|most common|most often", q.lower()):
        aggregation = "count"

    if intent is None:
        intent = "top_performers" if dimension else ("aggregate" if measure else "unsupported")

    # For a "most/top" ranking, additive volume/revenue wording should map to a
    # ranking over the selected measure; the ranking handler performs the grouping.
    if intent == "top_performers" and semantic_measure:
        measure = semantic_measure

    return Plan(
        intent=intent,
        measure=measure,
        dimension=dimension,
        second_measure=second,
        aggregation=aggregation,
        filters=_parse_filters(q, ds, measure),
    )


_PLAN_PROMPT = """You map a business question onto one analysis from a fixed list. Return the plan only.

Intents: {intents}
Measures (numeric fields): {measures}
Dimensions (categorical fields): {dims}
Has time axis: {time}

Rules: use only the exact field names listed; leave a field null when the question does not name one;
use "explain_change" for why/what-drove questions; "bottom_performers" for underperformance questions;
"unsupported" if the question cannot be answered from these fields.

Question: {q}"""


def llm_plan(q: str, ds: Dataset) -> Optional[Plan]:
    if not llm.available():
        return None
    intents = ", ".join(Intent.__args__)  # type: ignore[attr-defined]
    p = llm.plan(_PLAN_PROMPT.format(intents=intents, measures=ds.all_measures, time=ds.has_time, q=q,
                                     dims=ds.categorical_dimensions + ds.entities), Plan)
    return p


def validate_plan(p: Plan, ds: Dataset) -> Plan:
    cols = set(ds.df.columns)
    if p.measure not in ds.all_measures:
        p.measure = None
    if p.second_measure not in ds.all_measures:
        p.second_measure = None
    if p.dimension not in cols or ds.cols[p.dimension].role in ("id", "text", "constant", "datetime"):
        p.dimension = None
    p.filters = [f for f in p.filters if f.column in cols]
    return p


# --------------------------------------------------------------------------------------
class Copilot:
    def __init__(self, ds: Dataset, base: Analysis):
        self.ds, self.base = ds, base
        self._cache: dict[str, tuple[Dataset, Analysis]] = {}

    # ---- public --------------------------------------------------------------------
    def ask(self, question: str) -> Answer:
        q = (question or "").strip()
        if not q:
            return Answer("Ask a question about this dataset.", suggestions=self.suggestions())

        # Deterministic shortcut for direct record/ID retrieval. These requests do not
        # require an analytical model or LLM and should return exact values from the data.
        direct = self._try_record_lookup(q)
        if direct is not None:
            return direct
        try:
            self._q = q
            plan = self._plan(q)
            ds, A, note = self._scope(plan)
            ans = self._dispatch(plan, q, ds, A)
            if note:
                ans.basis = (note + " · " + ans.basis).strip(" ·")
        except Exception:
            ans = Answer("I couldn't complete that analysis.",
                         limitations=["Something in this question could not be evaluated against the available data. "
                                      "Try naming a specific measure or grouping field."],
                         suggestions=self.suggestions())
            return ans
        evidence = ans.evidence_text()
        if ans.facts and not ans.limitations:
            narrated = llm.interpret(q, evidence)
            if narrated:
                ans.interpretation = narrated
        return ans

    # ---- direct retrieval ---------------------------------------------------------
    def _try_record_lookup(self, q: str) -> Optional[Answer]:
        """Handle simple requests such as "give me some student IDs" exactly from the data."""
        ql = q.lower()
        if detect_intent(q) != "record_lookup":
            return None

        # Prefer explicit ID columns, then fall back to commonly used identifier names.
        id_cols = list(self.ds.id_columns)
        id_name_tokens = {
            "id", "identifier", "studentid", "student_no", "studentno", "roll",
            "rollno", "rollnumber", "registration", "registrationid", "registration_no",
            "admission", "admissionno", "admissionnumber", "enrollment", "enrollmentid",
            "enrollment_no", "customerid", "employeeid", "userid", "accountid",
        }
        for col in self.ds.df.columns:
            compact = re.sub(r"[^a-z0-9]", "", str(col).lower())
            toks = set(re.findall(r"[a-z0-9]+", str(col).lower()))
            if compact in id_name_tokens or toks & {"id", "identifier", "roll", "registration", "admission", "enrollment"}:
                if col not in id_cols:
                    id_cols.append(col)

        if not id_cols:
            return Answer(
                "No identifier column was detected.",
                limitations=["The dataset does not appear to contain a student/customer/employee ID field, so I cannot return IDs without guessing."],
            )

        # Choose the identifier that best matches the wording of the request.
        qt = _qtokens(q)
        scored = []
        for i, col in enumerate(id_cols):
            name = str(col).lower()
            nt = {_stem(t) for t in re.findall(r"[a-z0-9]+", name)}
            score = 2.0 * len(qt & nt)
            if "student" in ql and "student" in name:
                score += 8
            if "customer" in ql and "customer" in name:
                score += 8
            if "employee" in ql and "employee" in name:
                score += 8
            if "user" in ql and "user" in name:
                score += 8
            if "roll" in ql and "roll" in name:
                score += 8
            scored.append((score, -i, col))
        id_col = max(scored)[2]

        # "some" defaults to 10; explicit small numbers are respected.
        m = re.search(r"\b(?:top|first|some|show|give|list|display|fetch|provide|return|get)?\s*(\d{1,2})\b", ql)
        n = int(m.group(1)) if m else 10
        n = max(1, min(n, 20))

        vals = self.ds.df[id_col].dropna().drop_duplicates().head(n)
        if vals.empty:
            return Answer(f"No usable {self.ds.label(id_col)} values were found.",
                          limitations=[f"The {self.ds.label(id_col)} field contains no non-empty values."])

        label = self.ds.label(id_col)
        table = pd.DataFrame({label: vals.astype(str).tolist()})
        a = Answer(
            f"Here are {len(table)} {label.lower()} values",
            facts=[f"Showing {len(table)} distinct {label} values from the dataset."],
            basis=f"Source field: {label} · {len(self.ds.df):,} records",
            tables=[(f"Sample {label} values", table)],
        )
        return a

    def suggestions(self) -> list[str]:
        ds, A = self.ds, self.base
        P = ds.label(ds.primary) if ds.primary else "the main measure"
        qs = ["Create an executive summary.", "What are the most important findings?", "What should management investigate?"]
        if ds.has_time and ds.primary:
            qs.insert(1, f"Why did {P} change recently?")
            qs.append(f"What changed compared with the previous period?")
        if ds.categorical_dimensions and ds.primary:
            qs.append(f"Which {ds.label(ds.categorical_dimensions[0])} is underperforming?")
        qs.append("Which records look suspicious?")
        qs.append("What are the strongest potential drivers?")
        return qs[:8]

    # ---- planning ------------------------------------------------------------------
    def _plan(self, q: str) -> Plan:
        rp = rule_plan(q, self.ds)
        lp = llm_plan(q, self.ds)
        if lp is not None:
            lp = validate_plan(lp, self.ds)

            # Treat deterministic semantic signals as guardrails. The LLM may enrich a
            # plan, but it must not override an explicit business meaning such as
            # "sold" → quantity/volume or "revenue" → revenue.
            intent = rp.intent if rp.intent != "unsupported" else lp.intent
            measure = rp.measure or lp.measure
            dimension = rp.dimension or lp.dimension
            aggregation = rp.aggregation or lp.aggregation
            filters = rp.filters or lp.filters

            return validate_plan(Plan(
                intent=intent,
                measure=measure,
                dimension=dimension,
                second_measure=rp.second_measure or lp.second_measure,
                aggregation=aggregation,
                filters=filters,
                top_n=rp.top_n if rp.top_n != 10 else lp.top_n,
            ), self.ds)

        return validate_plan(rp, self.ds)

    def _scope(self, plan: Plan) -> tuple[Dataset, Analysis, str]:
        if not plan.filters:
            return self.ds, self.base, ""
        mask = pd.Series(True, index=self.ds.df.index)
        desc = []
        for f in plan.filters:
            c = self.ds.cols[f.column]
            if f.operator == "eq" and c.role not in ("measure",):
                mask &= self.ds.dim(f.column).str.lower() == str(f.value).lower()
                desc.append(f"{c.label} = {f.value}")
            else:
                v = self.ds.values(f.column)
                fv = float(f.value)
                mask &= {"eq": v == fv, "gt": v > fv, "gte": v >= fv, "lt": v < fv, "lte": v <= fv}[f.operator]
                sym = {"eq": "=", "gt": ">", "gte": "≥", "lt": "<", "lte": "≤"}[f.operator]
                desc.append(f"{c.label} {sym} {num(fv)}")
        if mask.sum() < 5:
            raise ValueError("filter too narrow")
        key = "|".join(desc)
        if key not in self._cache:
            sub = self.ds.subset(mask)
            self._cache[key] = (sub, analyze(sub) if plan.intent not in ("aggregate", "distribution") else None)
        sub, A = self._cache[key]
        if A is None and plan.intent not in ("aggregate", "distribution"):
            A = analyze(sub)
            self._cache[key] = (sub, A)
        return sub, A, f"Filtered to {', '.join(desc)} ({int(mask.sum()):,} records)"

    _CHANGE_WORDS = re.compile(r"declin|drop|fell|fall|decreas|lower|down|worse|dip|slump|shrink|increas|rise|rose|grew|growth|up\b|spike|higher|jump|improv|chang|move|shift|differ|happen|trend|perform|result|number|figure|metric|kpi")

    def _dispatch(self, plan: Plan, q: str, ds: Dataset, A: Optional[Analysis]) -> Answer:
        intent = plan.intent
        if intent == "explain_change" and not (plan.measure or plan.dimension or self._CHANGE_WORDS.search(q.lower())):
            return self._unsupported(ds)
        if intent in _NEEDS_TIME and not ds.has_time:
            return self._no_time(intent, ds)
        missing = self._missing_concept(q, plan, ds)
        if missing and intent not in ("executive_summary", "key_findings", "investigate", "overview", "data_quality",
                                      "anomalies", "segmentation", "risks", "opportunities"):
            return missing
        h = getattr(self, f"_h_{intent}", None)
        if h is None:
            return self._unsupported(ds)
        return h(plan, q, ds, A)

    # ---- guards --------------------------------------------------------------------
    def _no_time(self, intent: str, ds: Dataset) -> Answer:
        what = {"explain_change": "Change analysis", "period_comparison": "Period-over-period comparison",
                "trend": "Trend analysis", "forecast": "Forecasting", "seasonality": "Seasonality analysis"}[intent]
        alt = []
        if ds.categorical_dimensions and ds.primary:
            alt.append(f"Which {ds.label(ds.categorical_dimensions[0])} has the highest {ds.label(ds.primary)}?")
        alt.append("What are the strongest potential drivers?")
        return Answer(f"{what} is not available for this dataset.",
                      limitations=[f"{what} is unavailable because no suitable time/date column was detected."],
                      suggestions=alt)

    def _missing_concept(self, q: str, plan: Plan, ds: Dataset) -> Optional[Answer]:
        qt = {_stem(t) for t in re.findall(r"[a-z0-9]+", q.lower())}
        asked = {c for c in _CONCEPTS if c in qt}
        if not asked or plan.measure:
            return None
        names = " ".join(ds.df.columns).lower()
        for c in asked:
            related = {c} | _SYN.get(c, set())
            if any(r in names for r in related):
                return None
        concept = sorted(asked)[0]
        avail = join_and([ds.label(m) for m in ds.all_measures[:6]]) or "none"
        return Answer(f"The dataset has no field for “{concept}”.",
                      limitations=[f"No column corresponding to “{concept}” was found, so this cannot be answered without guessing. "
                                   f"Available numeric measures: {avail}."],
                      suggestions=[f"What is the average {ds.label(m)}?" for m in ds.all_measures[:2]])

    def _unsupported(self, ds: Dataset) -> Answer:
        return Answer("That question can't be answered from the available data.",
                      limitations=["The question doesn't map to an analysis this dataset supports. I can summarise performance, "
                                   "explain changes over time, rank segments, identify drivers, find unusual records and assess data quality."],
                      suggestions=self.suggestions())

    # ---- helpers -------------------------------------------------------------------
    def _measure(self, plan: Plan, ds: Dataset, A: Optional[Analysis]) -> str:
        return plan.measure or (A.primary if A else None) or ds.primary

    def _dimension(self, plan: Plan, ds: Dataset) -> tuple[Optional[str], bool]:
        if plan.dimension:
            return plan.dimension, False
        cands = ds.categorical_dimensions or ds.dimensions
        return (cands[0], True) if cands else (None, False)

    def _basis(self, ds: Dataset, measure: Optional[str], extra: str = "") -> str:
        bits = []
        if measure:
            bits.append(f"Measure: {ds.label(measure)}")
        bits.append(f"{len(ds.df):,} records")
        if extra:
            bits.append(extra)
        return " · ".join(bits)

    def _ts(self, ds: Dataset, A: Analysis, measure: str):
        if A is not None and measure == A.primary and A.ts is not None:
            return A.ts
        return tsm.build_ts(ds, measure)

    @staticmethod
    def _tbl(df: pd.DataFrame, formats: dict[str, Any]) -> pd.DataFrame:
        out = pd.DataFrame()
        for name, (col, fn) in formats.items():
            out[name] = df[col].map(fn)
        return out

    # ======================================================================================
    # Handlers
    # ======================================================================================
    def _h_executive_summary(self, plan, q, ds, A) -> Answer:
        a = Answer("Executive summary", facts=list(A.summary), basis=self._basis(ds, A.primary))
        a.tables.append(("Key findings", pd.DataFrame([{"Type": f.kind.title(), "Finding": f.title} for f in A.findings if f.importance >= 35][:8])))
        a.interpretation = self._interp_from_findings(A)
        a.recommendations = A.investigations[:5]
        if A.ts is not None and A.pop:
            a.figures.append(vz.trend_chart(ds, A.ts, A.forecast))
        return a

    def _interp_from_findings(self, A: Analysis) -> Optional[str]:
        notes = []
        if A.pop and A.change and A.change["dimensions"]:
            notes.append("The latest movement is localised rather than uniform, which makes it tractable to investigate.")
        if A.drivers and A.drivers.get("predictive"):
            notes.append("Other fields in the data explain a meaningful share of the primary measure, so segment-level "
                         "analysis is likely to be informative.")
        if A.quality["issues"]:
            notes.append("Data-quality issues should be resolved before figures are circulated externally.")
        return " ".join(notes) or None

    def _h_key_findings(self, plan, q, ds, A) -> Answer:
        top = [f for f in A.findings if f.importance >= 30][:8]
        a = Answer("Most important findings", basis=self._basis(ds, A.primary))
        a.facts = [f"**{f.title}.** {f.fact}" for f in top]
        a.recommendations = A.investigations[:4]
        a.interpretation = self._interp_from_findings(A)
        if not top:
            a.limitations = ["No material findings stood out in this dataset."]
        return a

    def _h_investigate(self, plan, q, ds, A) -> Answer:
        a = Answer("What management should investigate", basis=self._basis(ds, A.primary))
        pairs = [(f.title, f.investigate) for f in A.findings if f.investigate][:6]
        a.facts = [f"**{t}.** {next(x.fact for x in A.findings if x.title == t)}" for t, _ in pairs]
        a.recommendations = [i for _, i in pairs]
        if not pairs:
            a.limitations = ["No finding in this dataset is strong enough to recommend a specific investigation."]
        return a

    def _h_risks(self, plan, q, ds, A) -> Answer:
        r = A.risks[:6]
        a = Answer("Risks and areas requiring attention", basis=self._basis(ds, A.primary))
        a.facts = [f"**{f.title}.** {f.fact}" for f in r] or ["No material risks were identified by the available analyses."]
        a.recommendations = [f.investigate for f in r if f.investigate][:4]
        return a

    def _h_opportunities(self, plan, q, ds, A) -> Answer:
        o = A.opportunities[:6] + A.positives[:2]
        a = Answer("Opportunities and areas worth investigating", basis=self._basis(ds, A.primary))
        a.facts = [f"**{f.title}.** {f.fact}" for f in o] or ["No clear opportunities were identified by the available analyses."]
        a.recommendations = [f.investigate for f in o if f.investigate][:4]
        return a

    def _h_overview(self, plan, q, ds, A) -> Answer:
        roles = {}
        for c in ds.cols.values():
            roles.setdefault(c.role, []).append(c.label)
        a = Answer("Dataset overview", basis=self._basis(ds, A.primary))
        a.facts = [f"{len(ds.df):,} records and {ds.df.shape[1]} fields. Data health: {A.quality['score']:.0f}/100 ({A.quality['band'].lower()})."]
        if ds.measures or ds.rate_measures:
            a.facts.append("Measures: " + join_and([ds.label(m) for m in ds.all_measures]) + f". Primary measure: {ds.label(ds.primary)}.")
        if ds.categorical_dimensions:
            a.facts.append("Segmenting fields: " + join_and([ds.label(d) for d in ds.categorical_dimensions]) + ".")
        rng = ds.date_range()
        a.facts.append(f"Time coverage: {rng[0]:%d %b %Y} to {rng[1]:%d %b %Y} (by {ds.time_spec['label']})." if rng
                       else "No usable time field was detected, so time-based analyses are unavailable.")
        return a

    def _h_aggregate(self, plan, q, ds, A) -> Answer:
        M = self._measure(plan, ds, A)
        op = plan.aggregation or "average"
        if op == "count" or M is None:
            ent = plan.dimension if plan.dimension in ds.entities else None
            n = int(ds.df[ent].nunique()) if ent else len(ds.df)
            total = len(self.ds.df)
            noun = f"distinct {ds.label(ent)} values" if ent else "records"
            a = Answer(f"{n:,} {noun}", basis=self._basis(ds, None))
            a.facts = [f"{n:,} {noun} match" + (f" ({n / total * 100:.1f}% of {total:,} records)." if len(ds.df) < total and not ent else ".")
                       if len(ds.df) < total else f"There are {n:,} {noun} in the dataset."]
            return a
        kind, L = ds.kind(M), ds.label(M)
        v = ds.values(M).dropna()
        if v.empty:
            return Answer(f"No usable {L} values", limitations=[f"{L} has no usable numeric values."])
        fn = {"average": v.mean(), "sum": v.sum(), "median": v.median(), "minimum": v.min(), "maximum": v.max()}[op]
        word = {"average": "average", "sum": "total", "median": "median", "minimum": "minimum", "maximum": "maximum"}[op]
        a = Answer(f"The {word} {L} is {value(kind, fn)}", basis=self._basis(ds, M))
        if kind == "rate":
            a.facts.append(f"{L} is {value(kind, v.mean())} across {len(v):,} records.")
        else:
            a.facts.append(f"The {word} {L} is **{value(kind, fn)}** across {len(v):,} records "
                           f"(median {num(v.median())}; typical range {num(v.quantile(.25))}–{num(v.quantile(.75))}; "
                           f"overall {num(v.min())}–{num(v.max())}).")
            if v.median() and abs(v.mean() - v.median()) / abs(v.median()) >= 0.15:
                a.interpretation = (f"The average and median differ by {abs(v.mean() / v.median() - 1) * 100:.0f}%, so a minority of "
                                    f"{'high' if v.mean() > v.median() else 'low'} values is pulling the average; the median is the better "
                                    f"description of a typical record.")
            a.figures.append(vz.histogram(ds, M))
        dim = plan.dimension if plan.dimension and ds.cols[plan.dimension].role != "id" else None
        if dim:
            t = pf.group_table(ds, dim, M)
            col = {"sum": "total", "average": "avg"}.get(op, "avg")
            t = t[t["group"] != "(missing)"].sort_values(col, ascending=False).head(plan.top_n)
            a.tables.append((f"{word.title()} {L} by {ds.label(dim)}", self._tbl(t, {
                ds.label(dim): ("group", str), "Records": ("records", lambda x: f"{int(x):,}"),
                f"{word.title()} {L}": (col, lambda x: value(kind, x))})))
            if len(t):
                a.facts.append(f"By {ds.label(dim)}, {t.iloc[0]['group']} is highest ({value(kind, t.iloc[0][col])}) and "
                               f"{t.iloc[-1]['group']} lowest among the top {len(t)} ({value(kind, t.iloc[-1][col])}).")
        return a

    def _h_distribution(self, plan, q, ds, A) -> Answer:
        M = self._measure(plan, ds, A)
        d = A.distributions.get(M) if A else None
        if M is None or not d:
            return Answer("Distribution unavailable", limitations=["No numeric measure with enough values was found."])
        L = ds.label(M)
        a = Answer(f"Distribution of {L}", basis=self._basis(ds, M))
        a.facts = [f"{L}: median {num(d['median'])}, average {num(d['mean'])}, middle 50% between {num(d['p25'])} and {num(d['p75'])}, "
                   f"5th–95th percentile {num(d['p05'])}–{num(d['p95'])}, full range {num(d['min'])}–{num(d['max'])}."]
        if abs(d["skew"]) > 1:
            a.interpretation = (f"The distribution is {'right' if d['skew'] > 0 else 'left'}-skewed (skewness {d['skew']:.1f}); "
                                f"a small number of {'high' if d['skew'] > 0 else 'low'} values stretches the range.")
        a.figures.append(vz.histogram(ds, M))
        return a

    # ---- rankings ---------------------------------------------------------------------
    def _rank_common(self, plan, ds, A, lowest: bool) -> Answer:
        M = self._measure(plan, ds, A) if plan.aggregation != "count" else None
        dim, assumed = self._dimension(plan, ds)
        if dim is None:
            return Answer("Ranking unavailable", limitations=["A grouping field is required for ranking."])

        # Frequency questions are row-count rankings, not measure rankings.
        if plan.aggregation == "count":
            counts = ds.df[dim].value_counts(dropna=False)
            if lowest:
                counts = counts.sort_values(ascending=True)
            else:
                counts = counts.sort_values(ascending=False)
            counts = counts.head(plan.top_n)
            if counts.empty:
                return Answer("Frequency ranking unavailable", limitations=[f"No usable values were found for {ds.label(dim)}."])
            DL = ds.label(dim)
            r0 = counts.iloc[0]
            group = counts.index[0]
            word = "least frequent" if lowest else "most frequent"
            a = Answer(f"{group} is the {word} {DL}", basis=self._basis(ds, None, f"Count of records by {DL}"))
            a.facts.append(f"{group} appears {int(r0):,} times in the dataset.")
            table = pd.DataFrame({DL: counts.index.astype(str), "Records": counts.astype(int).values})
            a.tables.append((f"{word.title()} {DL} by record count", table))
            a.figures.append(vz.ranking_bar(table.rename(columns={DL: "group", "Records": "value"}), "group", "value",
                                            f"{word.title()} {DL} by record count", vz.RED if lowest else vz.BLUE, fmt_kind="intensity"))
            return a

        if M is None:
            return Answer("Ranking unavailable", limitations=["A numeric measure and a grouping field are both required for value-based ranking."])
        kind, L, DL = ds.kind(M), ds.label(M), ds.label(dim)
        r = pf.rank_groups(ds, dim, M, lowest=lowest, top_n=plan.top_n)
        if r.empty:
            return Answer("Ranking unavailable", limitations=[f"{L} has no usable values by {DL}."])
        word = "lowest" if lowest else "highest"
        pol = ds.polarity(M)
        basis = self._basis(ds, M, f"Grouped by {DL}" + (" (assumed)" if assumed else ""))
        a = Answer(f"{r.iloc[0]['group']} has the {word} {L} by {DL}", basis=basis)
        r0 = r.iloc[0]
        a.facts.append(f"By {DL}, **{r0['group']}** has the {word} {L}: {value(kind, r0['value'])}"
                       + (f" ({r0['share']:.0f}% of the total)" if kind == "additive" else f" across {int(r0['records']):,} records")
                       + (f", versus an overall {'total' if kind == 'additive' else 'average'} of "
                          f"{value(kind, headline_value(ds, M)[0])}." if kind != "additive" else "."))
        if kind == "additive":
            a.facts.append(f"Per record, {r0['group']} averages {num(r0['avg'])} ({r0['index']:.2f}x the overall per-record average), "
                           f"on {r0['record_share']:.0f}% of records — "
                           + ("its position reflects lower volume rather than lower value per record."
                              if (lowest and r0['index'] >= 0.95) else
                              "its position reflects value per record as well as volume." if abs(r0['index'] - 1) >= 0.1 else
                              "differences are driven mainly by volume, not value per record."))
        a.tables.append((f"{word.title()} {DL} by {L}", self._tbl(r, {
            DL: ("group", str), "Records": ("records", lambda x: f"{int(x):,}"),
            L: ("value", lambda x: value(kind, x)),
            "Share": ("share", lambda x: pct(x))})))
        a.figures.append(vz.ranking_bar(r, "group", "value", f"{word.title()} {DL} by {L}", vz.RED if lowest else vz.BLUE,
                                        fmt_kind=kind))
        # momentum: who is actually deteriorating?
        if ds.has_time and A is not None and M == A.primary and A.change:
            mv = self._movement(ds, A, M, dim)
            if mv is not None and len(mv) >= 2:
                net_pct = A.pop["pct"]
                worst = mv.sort_values("pct_change").iloc[0] if (A.pop["change"] < 0 or lowest) else mv.sort_values("pct_change").iloc[-1]
                if not pd.isna(worst["pct_change"]):
                    a.facts.append(f"Recent momentum ({A.pop['prev_label']} → {A.pop['latest_label']}): {worst['group']} "
                                   f"changed {pct(worst['pct_change'], signed=True)} versus {pct(net_pct, signed=True)} overall.")
                    if lowest and worst["group"] != r0["group"]:
                        if re.search(r"underperform|under-perform|struggl|declin|weak", getattr(self, "_q", "").lower()):
                            a.headline = (f"{worst['group']} shows the weakest recent momentum; {r0['group']} is lowest on {L}")
                        a.interpretation = (f"{r0['group']} is lowest on level, but {worst['group']} shows the weakest recent momentum; "
                                            f"“underperforming” depends on whether level or trend matters more to the decision.")
        if lowest:
            gaps = pf.segment_gaps(ds, dim, M)
            if not gaps.empty:
                g = gaps[gaps["group"] == r0["group"]]
                if len(g) and g.iloc[0]["p_value"] < 0.01:
                    a.facts.append(f"{r0['group']}'s per-record difference from the rest is statistically significant (p<0.01).")
            if pol >= 0:
                a.recommendations.append(f"Check whether {r0['group']} ({DL}) is structurally different (price level, mix, size) before treating it as underperformance.")
        pol = ds.polarity(M)
        adverse = pol != 0 and ((not lowest and pol < 0) or (lowest and pol > 0))
        favourable = pol != 0 and not adverse
        if not lowest or pol < 0:
            if adverse:
                a.recommendations.append(f"Investigate what is driving the weak {L} in {ds.seg_text(dim, r0['group'])} compared with other {DL} values.")
            elif favourable:
                a.recommendations.append(f"Identify what distinguishes {ds.seg_text(dim, r0['group'])} and whether it can be replicated elsewhere.")
            else:
                a.recommendations.append(f"Compare {ds.seg_text(dim, r0['group'])} with other {DL} values to understand what distinguishes it.")
        return a

    def _movement(self, ds: Dataset, A: Analysis, M: str, dim: str):
        if A.ts is None or A.pop is None:
            return None
        for d in (A.change or {}).get("dimensions", []):
            if d["dim"] == dim:
                return d["known"]
        m0, m1 = pf.period_masks(ds, A.ts, A.pop)
        t = pf.decompose_change(ds, M, dim, m0, m1)
        return None if t is None else t[t["group"] != "(missing)"]

    def _h_top_performers(self, plan, q, ds, A) -> Answer:
        return self._rank_common(plan, ds, A, lowest=False)

    def _h_bottom_performers(self, plan, q, ds, A) -> Answer:
        return self._rank_common(plan, ds, A, lowest=True)

    def _h_contribution(self, plan, q, ds, A) -> Answer:
        M = self._measure(plan, ds, A)
        dim, assumed = self._dimension(plan, ds)
        if M is None or dim is None:
            return Answer("Contribution unavailable", limitations=["A measure and a grouping field are required."])
        con = pf.contribution(ds, dim, M)
        if con is None:
            return Answer("Contribution unavailable", limitations=[f"{ds.label(dim)} has fewer than two groups."])
        L, DL, kind = ds.label(M), ds.label(dim), ds.kind(M)
        t = con["table"]
        basis_word = L if con["basis"] == "total" else "record count"
        a = Answer(f"{con['top1']} contributes the most {basis_word} by {DL}",
                   basis=self._basis(ds, M, f"Grouped by {DL}" + (" (assumed)" if assumed else "")))
        a.facts.append(f"**{con['top1']}** contributes the most {basis_word} across {DL}: {con['top1_share']:.1f}% of the total; "
                       f"the top three together contribute {con['top3_share']:.1f}%.")
        a.facts.append(f"{con['pareto_groups']} of {con['groups']} {DL} values account for 80% of {basis_word}.")
        if kind != "additive":
            a.limitations.append(f"{L} is an average-type measure, so contribution is shown as share of records rather than share of a total.")
        a.tables.append((f"Contribution by {DL}", self._tbl(t.head(plan.top_n), {
            DL: ("group", str), "Share": ("share", lambda x: pct(x)), "Records": ("records", lambda x: f"{int(x):,}"),
            f"Avg {L}": ("avg", lambda x: value(kind, x))})))
        a.figures.append(vz.pareto(t, f"Contribution to {basis_word} by {DL}"))
        a.interpretation = ("Results are highly concentrated, which creates dependency risk." if con["hhi_norm"] >= 0.15
                            else "Contribution is relatively evenly spread across groups.")
        if con["hhi_norm"] >= 0.15:
            a.recommendations.append(f"Assess dependency on {con['top1']} ({DL}) and how results would change if it underperformed.")
        return a

    # ---- change & time ------------------------------------------------------------------
    def _h_explain_change(self, plan, q, ds, A) -> Answer:
        M = self._measure(plan, ds, A)
        ts = self._ts(ds, A, M)
        if ts is None or len(ts.values.dropna()) < 2:
            return Answer("Not enough history", limitations=["At least two complete time periods are needed to explain a change."])
        ex = A.change if (A is not None and M == A.primary and ts is A.ts and A.change) else pf.explain_change(ds, M, ts)
        if ex is None:
            return Answer("Not enough history", limitations=["At least two complete time periods are needed to explain a change."])
        pop, kind, L = ex["pop"], ex["kind"], ds.label(M)
        pc, move = pop["pct"], ("fell" if pop["change"] < 0 else "rose")
        unit = unit_word(ts.freq)
        a = Answer("", basis=self._basis(ds, M, f"{pop['latest_label']} vs {pop['prev_label']}"))
        asked_decline = bool(re.search(r"declin|drop|fell|fall|decreas|lower|down|worse|dip|slump|shrink", q.lower()))
        asked_rise = bool(re.search(r"increas|rise|rose|grew|growth|\bup\b|spike|higher|jump|improv", q.lower()))
        if pc is not None and abs(pc) < 1:
            a.headline = f"{L} was essentially flat in {pop['latest_label']}"
            a.facts.append(f"{L} moved {pct(pc, signed=True)} ({value(kind, pop['prev'])} → {value(kind, pop['latest'])}); there is no meaningful change to explain.")
            return a
        if (asked_decline and pop["change"] > 0) or (asked_rise and not asked_decline and pop["change"] < 0):
            a.facts.append(f"**Premise check:** {L} did not {'decline' if asked_decline else 'rise'} — it {move} in the latest period. "
                           f"The analysis below explains the actual movement.")
        a.headline = f"{L} {move} {abs(pc):.1f}% in {pop['latest_label']}" if pc is not None else f"{L} changed in {pop['latest_label']}"
        a.facts.append(f"{L} {move} {abs(pc):.1f}% to {value(kind, pop['latest'])} in {pop['latest_label']}, from {value(kind, pop['prev'])} in "
                       f"{pop['prev_label']} (a change of {value(kind, pop['change'])})." if pc is not None else
                       f"{L} was {value(kind, pop['latest'])} versus {value(kind, pop['prev'])} previously.")
        if pop["z"] is not None:
            if pop["unusual"]:
                a.facts.append(f"Against past period-to-period changes, this move is **unusually large** (z={pop['z']:.1f}).")
            elif pc is not None and abs(pc) >= 20:
                a.facts.append(f"Swings of this size have occurred before in this series (z={pop['z']:.1f}), so size alone is not unusual; "
                               f"the year-over-year comparison is the better benchmark.")
            else:
                a.facts.append(f"This is within the normal range of past period-to-period changes (z={pop['z']:.1f}).")
        if pop["yoy_pct"] is not None:
            a.facts.append(f"Versus the same {unit} a year earlier ({pop['yoy_label']}), {L} is {pct(pop['yoy_pct'], signed=True)}.")
        if ex.get("p_value") is not None:
            a.facts.append(f"The difference in average {L} between the two periods is "
                           f"{'statistically significant' if ex['p_value'] < 0.05 else 'not statistically significant'} (p={ex['p_value']:.3f}).")
        if ts.partial_last:
            a.facts.append(f"{ts.label(ts.partial_last['start'])} is incomplete ({ts.partial_last['coverage'] * 100:.0f}% of the period) and was excluded.")
        vv = ex.get("volume_value")
        if vv:
            a.facts.append(f"**Volume vs value:** records went from {vv['n0']:,} to {vv['n1']:,} (volume effect {num(vv['volume_effect'])}) while "
                           f"average {L} per record went from {num(vv['avg0'])} to {num(vv['avg1'])} (value effect {num(vv['value_effect'])}).")
        net = pop["change"]
        for d in ex["dimensions"][:3]:
            lead, DL = d["lead"], ds.label(d["dim"])
            share = abs(lead["share_of_change"]) if not pd.isna(lead["share_of_change"]) else None
            line = f"**By {DL}:** {ds.seg_short(d['dim'], lead['group'])} stands out"
            if share is not None:
                line += f", accounting for {share:.0f}% of the net change"
                if not pd.isna(lead.get("base_share")) and kind == "additive":
                    line += f" from {lead['base_share']:.0f}% of the prior-period base"
            if not pd.isna(lead.get("pct_change")):
                line += f" ({pct(lead['pct_change'], signed=True)} versus {pct(pc, signed=True)} overall)"
            a.facts.append(line + ".")
            if abs(d["unattributed"]) >= 0.05 * abs(net):
                a.facts.append(f"Records with a missing {DL} account for {num(d['unattributed'])} of the change and cannot be attributed.")
        if ex["pairs"]:
            pr = ex["pairs"][0]
            top = pr["table"].iloc[0]
            a.facts.append(f"**Most concentrated combination:** {top['group']} ({' × '.join(ds.label(x) for x in pr['dims'])}) "
                           f"changed {num(top['delta'])}, {abs(top['share_of_change']):.0f}% of the net change.")
        infl = ex.get("influence")
        if infl and max(infl.get("prev_top_share") or 0, infl.get("latest_top_share") or 0) >= 15:
            a.facts.append(f"A single record represents {max(infl['prev_top_share'] or 0, infl['latest_top_share'] or 0):.0f}% of one period's total, "
                           f"so the comparison is sensitive to individual records.")
        # tables + charts
        if ex["dimensions"]:
            d0 = ex["dimensions"][0]
            t = d0["table"].head(8)
            cols = {ds.label(d0["dim"]): ("group", str), f"{pop['prev_label']}": ("v0", lambda x: value(kind, x)),
                    f"{pop['latest_label']}": ("v1", lambda x: value(kind, x)), "Change": ("delta", lambda x: value(kind, x)),
                    "% change": ("pct_change", lambda x: pct(x, signed=True)),
                    "Share of change": ("share_of_change", lambda x: pct(x))}
            a.tables.append((f"{L} change by {ds.label(d0['dim'])}", self._tbl(t, cols)))
            if kind == "additive":
                a.figures.append(vz.waterfall(d0["table"], pop, L, ds.label(d0["dim"])))
        a.figures.append(vz.trend_chart(ds, ts, None))
        # interpretation (deterministic) & recommendations (rule-based)
        d0 = ex["dimensions"][0] if ex["dimensions"] else None
        if d0:
            same = (np.sign(d0["known"]["delta"]) == np.sign(net)).mean()
            spread = "broad-based — most groups moved in the same direction" if same >= 0.75 else "concentrated in specific groups"
            driver = ""
            if vv:
                driver = (" Most of the movement comes from " +
                          ("fewer/more records (volume)" if abs(vv["volume_effect"]) > abs(vv["value_effect"]) else "a change in average value per record") + ".")
            a.interpretation = (f"The {move.replace('fell', 'decline').replace('rose', 'increase')} is {spread}, with "
                                f"{ds.seg_short(d0['dim'], d0['lead']['group'])} moving most out of line.{driver} "
                                f"This shows where the change occurred; the data alone cannot establish the operational cause.")
            a.recommendations.append(f"Investigate {ds.seg_text(d0['dim'], d0['lead']['group'])} between {pop['prev_label']} and {pop['latest_label']}.")
            if vv:
                a.recommendations.append("Review " + ("demand, capacity and data completeness for the change in record volume." if abs(vv["volume_effect"]) > abs(vv["value_effect"])
                                                    else "pricing, discounting and product mix, which determine average value per record."))
            if ex["pairs"]:
                a.recommendations.append(f"Drill into {ex['pairs'][0]['table'].iloc[0]['group']}, the most concentrated combination.")
        return a

    def _h_period_comparison(self, plan, q, ds, A) -> Answer:
        ms = [plan.measure] if plan.measure else ([A.primary] + [m for m in ds.all_measures if m != A.primary][:3])
        rows, facts, label = [], [], None
        for m in ms:
            ts = self._ts(ds, A, m)
            p = tsm.period_over_period(ts) if ts is not None else None
            if not p:
                continue
            label = (p["latest_label"], p["prev_label"])
            kind = ds.kind(m)
            rows.append({"Measure": ds.label(m), label[1]: value(kind, p["prev"]), label[0]: value(kind, p["latest"]),
                         "Change": pct(p["pct"], signed=True) if p["pct"] is not None else "n/a",
                         "vs last year": pct(p["yoy_pct"], signed=True) if p["yoy_pct"] is not None else "n/a"})
            if p["pct"] is not None:
                facts.append(f"{ds.label(m)} {'rose' if p['pct'] > 0 else 'fell'} {abs(p['pct']):.1f}% ({value(kind, p['prev'])} → {value(kind, p['latest'])})"
                             + (", an unusually large move." if p["unusual"] else "."))
        if not rows:
            return Answer("No comparison available", limitations=["At least two complete time periods are required for a comparison."])
        a = Answer(f"Change from {label[1]} to {label[0]}", facts=facts, tables=[("Period comparison", pd.DataFrame(rows))],
                   basis=self._basis(ds, None, f"{label[0]} vs {label[1]}"))
        if A.ts is not None and A.ts.partial_last:
            a.facts.append(f"{A.ts.label(A.ts.partial_last['start'])} is incomplete and was excluded from the comparison.")
        M0 = plan.measure or A.primary
        if M0 == A.primary and A.change and A.change["dimensions"] and A.pop and A.pop["pct"] is not None and abs(A.pop["pct"]) >= 5:
            d0 = A.change["dimensions"][0]
            a.facts.append(f"The largest out-of-line mover for {ds.label(M0)} is {ds.seg_text(d0['dim'], d0['lead']['group'])} "
                           f"({pct(d0['lead']['pct_change'], signed=True)}); ask “why did {ds.label(M0)} change?” for the full attribution.")
        if ds.has_time and A.ts is not None:
            a.figures.append(vz.trend_chart(ds, A.ts, None))
        return a

    def _h_trend(self, plan, q, ds, A) -> Answer:
        M = self._measure(plan, ds, A)
        ts = self._ts(ds, A, M)
        if ts is None or len(ts.values.dropna()) < 6:
            return Answer("Trend unavailable", limitations=["Trend analysis needs at least 6 complete time periods."])
        tr, L, kind, u = tsm.trend(ts), ds.label(M), ds.kind(M), unit_word(ts.freq)
        a = Answer(f"{L} over time", basis=self._basis(ds, M, f"{len(ts.values)} {u}s"))
        v = ts.values
        a.facts.append(f"{L} ran from {value(kind, v.iloc[0])} ({ts.label(v.index[0])}) to {value(kind, v.iloc[-1])} ({ts.label(v.index[-1])}); "
                       f"peak {value(kind, v.max())} in {ts.label(v.idxmax())}, low {value(kind, v.min())} in {ts.label(v.idxmin())}.")
        if tr:
            if tr["significant"]:
                a.facts.append(f"There is a statistically significant {'upward' if tr['direction'] == 'up' else 'downward'} trend of about "
                               f"{abs(tr['pct_per_period']):.1f}% per {u} (p={tr['p_value']:.3f}).")
            else:
                a.facts.append(f"There is no statistically significant linear trend (p={tr['p_value']:.2f}); movements are better described as fluctuation.")
        se = tsm.seasonality(ts)
        if se and se["meaningful"]:
            a.facts.append(f"The series is seasonal (strength {se['strength'] * 100:.0f}%), peaking in {se['peak']} and lowest in {se['trough']}.")
        p = tsm.period_over_period(ts)
        if p and p["pct"] is not None:
            a.facts.append(f"Latest period ({p['latest_label']}): {pct(p['pct'], signed=True)} versus {p['prev_label']}.")
        if ts.partial_last:
            a.facts.append(f"{ts.label(ts.partial_last['start'])} is incomplete and was excluded.")
        a.figures.append(vz.trend_chart(ds, ts, A.forecast if (A and M == A.primary) else None))
        if tr and tr.get("volatility_cv") and tr["volatility_cv"] > 0.4:
            a.interpretation = f"{L} is volatile (variation of {tr['volatility_cv'] * 100:.0f}% of its average), so single-period moves should be read with caution."
        return a

    def _h_forecast(self, plan, q, ds, A) -> Answer:
        M = self._measure(plan, ds, A)
        ts = self._ts(ds, A, M)
        if ts is None:
            return Answer("Forecast unavailable", limitations=["Forecasting is unavailable because no usable time series could be built."])
        se = tsm.seasonality(ts)
        hz = self._horizon(q, ts.freq)
        fc = A.forecast if (A and M == A.primary and A.forecast and not hz) else tsm.forecast(ts, se, horizon=hz)
        if not fc.get("available"):
            return Answer("Forecast unavailable", limitations=[fc["reason"]], basis=self._basis(ds, M))
        L, kind, u = ds.label(M), ds.kind(M), unit_word(fc["freq"])
        f = fc["frame"]
        a = Answer(f"{L} forecast: next {fc['horizon']} {u}s", basis=self._basis(ds, M, f"{fc['history_periods']} {u}s of history"))
        a.facts.append(f"The {fc['model']} projects {L} to average **{value(kind, fc['future_mean'])}** per {u} over the next {fc['horizon']} {u}s, "
                       f"{pct(fc['change_pct'], signed=True)} versus the last {fc['horizon']}.")
        a.facts.append(f"Reliability is **{fc['reliability'].lower()}**: in a back-test on the last {fc['backtest_periods']} {u}s, average error was "
                       f"{pct((fc['backtest_mape'] or 0) * 100, decimals=0)}"
                       + (f", {fc['skill_vs_naive'] * 100:.0f}% better than a naive baseline." if fc.get("skill_vs_naive") is not None and fc["skill_vs_naive"] > 0 else "."))
        tb = pd.DataFrame({"Period": [ts.label(x) for x in f["period"]], "Forecast": f["forecast"].map(lambda x: value(kind, x)),
                           "Likely range (80%)": [f"{value(kind, lo)} – {value(kind, hi)}" for lo, hi in zip(f["lower"], f["upper"])]})
        a.tables.append(("Forecast", tb))
        a.figures.append(vz.trend_chart(ds, ts, fc, f"{L}: history and forecast"))
        if fc["reliability"] == "Low":
            a.limitations.append("Back-test error is high, so treat this forecast as indicative only.")
        a.interpretation = "The forecast assumes historical patterns continue; it does not account for planned actions or external events."
        return a

    @staticmethod
    def _horizon(q: str, freq: str) -> Optional[int]:
        m = re.search(r"next\s+(\d+\s+)?(day|week|month|quarter|year)s?", q.lower())
        if not m:
            return None
        n = int(m.group(1)) if m.group(1) else 1
        per_unit = {"M": {"month": 1, "quarter": 3, "year": 12}, "Q": {"quarter": 1, "year": 4},
                    "W": {"week": 1, "month": 4, "quarter": 13}, "D": {"day": 1, "week": 7, "month": 30},
                    "Y": {"year": 1}}.get(freq, {})
        k = per_unit.get(m.group(2))
        return n * k if k else None

    def _h_seasonality(self, plan, q, ds, A) -> Answer:
        M = self._measure(plan, ds, A)
        ts = self._ts(ds, A, M)
        se = tsm.seasonality(ts) if ts is not None else None
        L = ds.label(M)
        if not se:
            return Answer("Seasonality unavailable", limitations=[f"Seasonality analysis needs roughly 2+ full seasonal cycles of complete history (e.g. 24+ months); {L} does not have enough."])
        a = Answer(f"{L} seasonality", basis=self._basis(ds, M))
        if se["meaningful"]:
            a.facts.append(f"{L} shows a clear seasonal pattern (strength {se['strength'] * 100:.0f}%): highest in {se['peak']} "
                           f"({pct(se['peak_pct'], signed=True, decimals=0)} vs average), lowest in {se['trough']} ({pct(se['trough_pct'], signed=True, decimals=0)}).")
            a.recommendations.append(f"Align targets, capacity and campaigns with the {se['peak']} peak and {se['trough']} trough.")
        else:
            a.facts.append(f"No meaningful seasonal pattern was found (strength {se['strength'] * 100:.0f}%, below the 30% threshold).")
        a.figures.append(vz.seasonality_bar(se))
        return a

    # ---- drivers & structure ------------------------------------------------------------
    def _h_drivers(self, plan, q, ds, A) -> Answer:
        M = self._measure(plan, ds, A)
        from . import drivers as dr
        d = A.drivers if (A and M == A.primary and A.drivers) else dr.analyze_drivers(ds, M)
        L = ds.label(M)
        if not d.get("available"):
            return Answer("Driver analysis unavailable", limitations=[d["reason"]])
        a = Answer(f"Potential drivers of {L}", basis=self._basis(ds, M))
        met = "R²" if d["metric"] == "R2" else "AUC"
        if not d["predictive"] or not d["drivers"]:
            a.facts.append(f"No field in this dataset predicts {L} reliably (out-of-sample {met} {d['score']:.2f}).")
            a.limitations.append("Any single-factor explanation of this measure would be unreliable with the data available.")
            return a
        a.facts.append(f"Together, the other fields predict {L} out-of-sample with {met} {d['score']:.2f}. Strongest associated fields:")
        for x in d["drivers"]:
            if x["type"] == "numeric" and ds.cols[x["feature"]].role == "binary":
                pos = ds.cols[x["feature"]].positive
                pos_txt = "Yes" if pos in (1, True, "1") else str(pos)
                s = (f"**{x['label']}** ({x['importance_pct']:.0f}% of importance): records with {x['label']} = {pos_txt} have "
                     f"{'higher' if x['direction'] == 'positive' else 'lower'} {L} (rank correlation {x['rho']:+.2f})")
                if "low_quartile_mean" not in x:
                    s += "."
                else:
                    s += f": {value(ds.kind(M), x['high_quartile_mean'])} versus {value(ds.kind(M), x['low_quartile_mean'])} otherwise."
            elif x["type"] == "numeric":
                dirn = "higher" if x["direction"] == "positive" else "lower"
                s = f"**{x['label']}** ({x['importance_pct']:.0f}% of importance): higher values go with {dirn} {L} (rank correlation {x['rho']:+.2f})"
                if "low_quartile_mean" in x:
                    k = ds.kind(M)
                    s += f"; the lowest quartile averages {value(k, x['low_quartile_mean'])} versus {value(k, x['high_quartile_mean'])} in the highest."
            else:
                k = ds.kind(M)
                s = f"**{x['label']}** ({x['importance_pct']:.0f}% of importance)"
                if "best" in x:
                    s += f": {x['best']} averages {value(k, x['best_mean'])} versus {value(k, x['worst_mean'])} for {x['worst']}"
                s += "."
            a.facts.append(s)
        if d.get("redundant"):
            a.facts.append(f"Excluded as near-duplicates of {L}: {join_and([ds.label(c) for c in d['redundant']])}.")
        a.interpretation = ("These fields move with the measure but association is not causation; they identify where to look, not what to change.")
        a.recommendations.append(f"Validate {d['drivers'][0]['label']} with a controlled comparison before treating it as a lever.")
        a.figures.append(vz.driver_bars(d))
        return a

    def _h_correlation(self, plan, q, ds, A) -> Answer:
        if plan.measure and plan.second_measure:
            x, y = ds.values(plan.measure), ds.values(plan.second_measure)
            both = pd.concat([x, y], axis=1).dropna()
            if len(both) < 10:
                return Answer("Not enough data", limitations=["Too few records have both fields to measure a relationship."])
            r, p = stats.pearsonr(both.iloc[:, 0], both.iloc[:, 1])
            rs = stats.spearmanr(both.iloc[:, 0], both.iloc[:, 1]).statistic
            strength = "very strong" if abs(r) >= .8 else "strong" if abs(r) >= .6 else "moderate" if abs(r) >= .4 else "weak"
            a = Answer(f"{ds.label(plan.measure)} and {ds.label(plan.second_measure)}: {strength} relationship",
                       basis=self._basis(ds, plan.measure))
            a.facts.append(f"The correlation between {ds.label(plan.measure)} and {ds.label(plan.second_measure)} is {r:+.2f} "
                           f"(rank correlation {rs:+.2f}) across {len(both):,} records, {'statistically significant' if p < 0.01 else 'not clearly significant'}.")
            a.interpretation = "Correlation shows that the fields move together, not that one causes the other."
            return a
        c = A.corr
        if not c:
            return Answer("Correlations unavailable", limitations=["At least two numeric fields are needed."])
        a = Answer("Strongest relationships between numeric fields", basis=self._basis(ds, None))
        for p in c["pairs"][:5]:
            a.facts.append(f"{p['label_a']} and {p['label_b']}: r = {p['r']:+.2f}" + (" (relationship is non-linear)" if p["nonlinear"] else "") + ".")
        a.figures.append(vz.corr_heatmap(c["matrix"]))
        a.interpretation = "Correlation shows that fields move together, not that one causes the other."
        return a

    def _h_anomalies(self, plan, q, ds, A) -> Answer:
        an_ = A.anomalies
        if not an_.get("available"):
            return Answer("Anomaly detection unavailable", limitations=[an_.get("reason", "Not enough numeric data.")])
        a = Answer("Records that look unusual", basis=self._basis(ds, None))
        if not an_["total"]:
            a.facts.append("No records stood out as unusual using the available methods.")
            return a
        a.facts.append(f"**{an_['total']:,} records ({an_['share_pct']:.1f}%)** look unusual; {an_['high']:,} are high priority.")
        if an_["by_field"]:
            top = sorted(an_["by_field"].items(), key=lambda kv: -kv[1])[:3]
            a.facts.append("Most flagged fields: " + join_and([f"{k} ({v:,})" for k, v in top]) + ".")
        if A.quality["duplicate_rows"]:
            a.facts.append(f"Separately, {A.quality['duplicate_rows']:,} records are exact duplicates of other rows and may be double-counted.")
        for pa in A.period_anoms[:2]:
            a.facts.append(f"{pa['period']} was {pa['direction']} compared with its seasonal expectation ({num(pa['value'])} vs about {num(pa['expected'])}).")
        a.tables.append(("Highest-priority unusual records", an_["table"].head(10)))
        a.interpretation = "“Unusual” means statistically atypical; these may be genuine large events, entry errors, or misuse — only the data owner can confirm."
        a.recommendations.append("Review the high-priority records with the data owner and confirm whether each is genuine, an entry error or misuse.")
        return a

    def _h_valuable_entities(self, plan, q, ds, A) -> Answer:
        M = self._measure(plan, ds, A)
        ent = plan.dimension if plan.dimension in ds.entities else (ds.entities[0] if ds.entities else None)
        if ent is None:
            if M is None:
                return Answer("Unavailable", limitations=["No measure was found to rank records by value."])
            v = ds.values(M).dropna().sort_values(ascending=False)
            a = Answer(f"Highest-value records by {ds.label(M)}", basis=self._basis(ds, M))
            a.limitations.append("No customer-like identifier field was detected, so individual records are ranked instead.")
            a.facts.append(f"The top 10 records hold {v.head(10).sum() / v.sum() * 100:.1f}% of total {ds.label(M)}." if ds.kind(M) == "additive" else
                           f"The highest {ds.label(M)} is {num(v.iloc[0])}.")
            return a
        res = pf.valuable_entities(ds, ent, M)
        if res is None:
            return Answer("Unavailable", limitations=[f"{ds.label(ent)} has too few distinct values with enough records to compare."])
        L, EL, kind = ds.label(M), ds.label(ent), res["kind"]
        fl = res["flagged"]
        a = Answer(f"Unusually valuable {EL} values", basis=self._basis(ds, M, f"{res['entities']:,} {EL} values"))
        if len(fl):
            a.facts.append(f"**{len(fl)} {EL} value(s) are statistically unusual** on {L}: " +
                           join_and([f"{r['entity']} ({num(r['value'])}, {r['x_median']:.1f}x the typical {EL})" for _, r in fl.head(5).iterrows()]) + ".")
        else:
            a.facts.append(f"No {EL} stands out as statistically unusual on {L}; the highest are listed below.")
        if kind == "additive":
            a.facts.append(f"The top {len(res['table'])} {EL} values contribute {res['top_share']:.1f}% of total {L}; the single largest contributes {res['top1_share']:.1f}%.")
        a.tables.append((f"Top {EL} by {L}", self._tbl(res["table"], {EL: ("entity", str), "Records": ("records", lambda x: f"{int(x):,}"),
                                                                       L: ("value", lambda x: value(kind, x)),
                                                                       "× typical": ("x_median", lambda x: f"{x:.1f}x")})))
        a.interpretation = "Statistically unusual means far above the typical level for this field; it does not by itself indicate quality or profitability."
        a.recommendations.append(f"Review dependence on the top {EL} values and whether their activity is repeatable.")
        return a

    def _h_segmentation(self, plan, q, ds, A) -> Answer:
        s = A.segments
        if not s.get("available"):
            return Answer("No distinct segments found", limitations=[s["reason"]])
        L = ds.label(ds.primary) if ds.primary else ""
        a = Answer(f"{s['k']} statistically distinct segments", basis=self._basis(ds, ds.primary))
        a.facts.append(f"Records separate into {s['k']} groups (cluster separation {s['silhouette']:.2f}, where above 0.25 is meaningful).")
        rows = []
        for p in s["profiles"]:
            rows.append({"Segment": p["name"], "Records": f"{p['records']:,}", "Share": pct(p["share_pct"]),
                         "Characterised by": p["description"],
                         f"Avg {L}": value(ds.kind(ds.primary), p["primary_mean"]) if p.get("primary_mean") is not None else "n/a"})
        a.tables.append(("Segment profiles", pd.DataFrame(rows)))
        a.figures.append(vz.segment_scatter(s))
        a.interpretation = "Segments are statistical groupings of similar records, not predefined business personas."
        return a

    def _h_data_quality(self, plan, q, ds, A) -> Answer:
        qd = A.quality
        a = Answer(f"Data health: {qd['score']:.0f}/100 ({qd['band'].lower()})", basis=self._basis(ds, None))
        a.facts.append(f"Completeness {qd['completeness']:.0f}, uniqueness {qd['uniqueness']:.0f}, validity {qd['validity']:.0f} (each out of 100). "
                       f"{qd['missing_cells']:,} cells ({qd['missing_pct']:.1f}%) are missing and {qd['duplicate_rows']:,} rows are duplicated.")
        for i in qd["issues"][:5]:
            a.facts.append(f"**{i['title']}.** {i['detail']}")
            a.recommendations.append(i["action"])
        if not qd["issues"]:
            a.facts.append("No material data-quality issues were detected.")
        if (qd["columns_table"]["Missing %"] > 0).any():
            a.figures.append(vz.missing_bar(qd["columns_table"]))
        return a
