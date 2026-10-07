"""Dataset loading, type cleaning and semantic schema inference.

Nothing here assumes a particular business domain. Column roles (measure,
dimension, entity, identifier, time, flag) are inferred from data shape and
naming conventions, and every downstream analysis is driven by those roles.
"""
from __future__ import annotations

import io
import re
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from .fmt import pretty, tokens

MAX_DIM_CARDINALITY = 50

_BOOL_WORDS = {"yes", "no", "y", "n", "true", "false", "t", "f"}
_POS_BOOL = {"yes", "y", "true", "t", "1"}

_ID_TOKENS = {"id", "key", "uuid", "guid", "ref", "reference", "index", "idx", "no", "number", "num", "code", "sku"}
_CODE_TOKENS = {"zip", "zipcode", "postal", "pincode", "pin", "phone", "mobile", "tel", "lat", "latitude", "lon", "lng", "longitude"}
_YEAR_TOKENS = {"year", "yr", "fy", "fiscal"}
_MONTH_TOKENS = {"month", "mo", "mon", "mth"}
_DATE_NAME_TOKENS = {"date", "time", "timestamp", "datetime", "day", "period", "created", "updated", "dt", "week", "month"}
_DATE_PREFER = {"order", "transaction", "sale", "sold", "invoice", "purchase", "created", "booking", "event", "date"}
_DATE_AVOID = {"birth", "dob", "ship", "shipped", "delivery", "delivered", "due", "end", "expire", "expiry", "renewal", "hire", "join"}
_YEAR_AVOID = {"built", "birth", "founded", "manufactured", "model", "released", "release", "hire", "hired", "joined",
               "start", "graduation", "established", "construction", "remodel", "remodeled", "renovated", "vintage"}
_YEAR_PREFER = {"sold", "sale", "order", "transaction", "reporting", "fiscal", "fy", "period", "observation", "survey", "event"}

_ADDITIVE = {
    "revenue", "sales", "amount", "total", "quantity", "qty", "units", "volume", "profit", "cost", "expense",
    "expenses", "spend", "orders", "clicks", "impressions", "visits", "count", "turnover", "income", "earnings",
    "bookings", "claims", "tickets", "shipments", "shipment", "shipped", "boxes", "box", "items", "sessions", "payments", "budget",
}
_NOT_ADDITIVE = {"price", "rate", "avg", "average", "per", "ratio", "pct", "percent", "percentage", "score", "age",
                 "margin", "discount", "mean", "median", "index", "rating", "temperature", "density"}
_POSITIVE = {"revenue", "sales", "profit", "income", "earnings", "score", "rating", "satisfaction", "conversion",
             "orders", "quantity", "units", "bookings", "retention", "growth", "margin", "clicks", "visits",
             "sessions", "performance", "yield", "output", "turnover"}
_NEGATIVE = {"cost", "expense", "expenses", "churn", "attrition", "defect", "defects", "loss", "losses", "delay",
             "delays", "return", "returns", "refund", "refunds", "complaint", "complaints", "error", "errors",
             "fraud", "default", "cancel", "cancelled", "canceled", "late", "downtime", "waste", "absent",
             "absence", "risk", "failure", "failures", "bounce"}
_MEASURE_PRIORITY = [
    ("revenue", 10), ("sales", 10), ("turnover", 8), ("profit", 9), ("income", 7), ("amount", 7), ("price", 8),
    ("salary", 8), ("spend", 6), ("value", 6), ("cost", 5), ("total", 4), ("charge", 5), ("fare", 5), ("rent", 5),
    ("premium", 5), ("balance", 5), ("score", 3), ("quantity", 3), ("units", 3), ("volume", 3),
]


@dataclass
class Col:
    name: str
    role: str                    # measure | binary | dimension | entity | text | id | code | year | month | datetime | constant
    label: str
    kind: Optional[str] = None   # measures only: additive | intensity | rate
    polarity: int = 0            # +1 higher is better, -1 lower is better, 0 neutral/unknown
    discrete: bool = False
    positive: Any = None         # positive label for string binary columns
    nunique: int = 0


# --------------------------------------------------------------------------------------
# Loading & cleaning
# --------------------------------------------------------------------------------------
_NUM_RE = re.compile(r"^\s*[\(\-+]?\s*[$€£₹]?\s*\d[\d,]*\.?\d*\s*%?\s*\)?\s*$")


def _is_texty(s: pd.Series) -> bool:
    return (
        s.dtype == object
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    ) and not pd.api.types.is_bool_dtype(s)


def _coerce_numeric_strings(s: pd.Series) -> pd.Series:
    nn = s.dropna().astype(str)
    if len(nn) < 3:
        return s
    sample = nn.head(1000)
    if sample.map(lambda v: bool(_NUM_RE.match(v))).mean() < 0.95:
        return s
    if sample.str.match(r"^\s*0\d").any() or sample.str.replace(r"\D", "", regex=True).str.len().median() >= 10:
        return s  # zero-padded codes / phone-like numbers are identifiers, not measures
    cleaned = (
        s.astype("object").where(s.notna(), None).astype(str)
        .str.replace(r"[,$€£₹%\s]", "", regex=True)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    )
    out = pd.to_numeric(cleaned.where(s.notna()), errors="coerce")
    return out if out.notna().sum() >= 0.95 * s.notna().sum() else s


def _try_datetime(s: pd.Series, name: str) -> Optional[pd.Series]:
    nn = s.dropna().astype(str)
    if len(nn) < 4:
        return None
    sample = nn.sample(min(len(nn), 400), random_state=0)
    named_like_date = bool(set(tokens(name)) & _DATE_NAME_TOKENS)
    if sample.str.len().median() < 6 and not named_like_date:
        return None
    if sample.str.contains(r"\d").mean() < 0.9:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    if parsed.notna().mean() < 0.9 or parsed.nunique() < 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = pd.to_datetime(s, errors="coerce", format="mixed")
    return full


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = _unique_names([str(c) for c in df.columns])
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    for col in list(df.columns):
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            if getattr(s.dt, "tz", None) is not None:
                df[col] = s.dt.tz_localize(None)
            continue
        if _is_texty(s):
            s = s.astype("object")
            s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
            s = s.replace({"": np.nan, "NA": np.nan, "N/A": np.nan, "n/a": np.nan, "null": np.nan, "NULL": np.nan,
                           "None": np.nan, "nan": np.nan, "-": np.nan})
            numeric = _coerce_numeric_strings(s)
            if pd.api.types.is_numeric_dtype(numeric):
                df[col] = numeric
                continue
            as_date = _try_datetime(s, col)
            df[col] = as_date if as_date is not None else s
    return df.reset_index(drop=True)


def _unique_names(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for raw in columns:
        name = raw.strip() or "Unnamed"
        count = seen.get(name, 0)
        seen[name] = count + 1
        out.append(name if count == 0 else f"{name}_{count + 1}")
    return out


def load_data(source: Any, filename: str | None = None) -> pd.DataFrame:
    name = (filename or getattr(source, "name", "dataset.csv")).lower()
    if hasattr(source, "seek"):
        source.seek(0)
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        df = pd.read_excel(source)
    else:
        raw = source.read() if hasattr(source, "read") else open(source, "rb").read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        df = None
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=enc)
                break
            except UnicodeDecodeError:
                continue
            except Exception:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                break
        if df is None:
            raise ValueError("The file could not be decoded as a CSV.")
    return clean_frame(df)


# --------------------------------------------------------------------------------------
# Schema inference
# --------------------------------------------------------------------------------------
def _measure_kind(tok: set[str]) -> str:
    if tok & _NOT_ADDITIVE:
        return "intensity"
    return "additive" if tok & _ADDITIVE else "intensity"


def _polarity(tok: set[str]) -> int:
    if tok & _NEGATIVE:
        return -1
    if tok & _POSITIVE:
        return 1
    return 0


def _is_sequential_index(s: pd.Series) -> bool:
    if not pd.api.types.is_integer_dtype(s) or s.isna().any() or len(s) < 5:
        return False
    d = np.diff(s.to_numpy())
    return bool((d == 1).all() or (d == -1).all())


def infer_columns(df: pd.DataFrame) -> dict[str, Col]:
    n = len(df)
    cols: dict[str, Col] = {}
    for name in df.columns:
        s = df[name]
        tok = set(tokens(name))
        nun = int(s.nunique(dropna=True))
        info = Col(name=name, role="text", label=pretty(name), nunique=nun)
        ratio = nun / max(n, 1)

        if nun <= 1:
            info.role = "constant"
        elif pd.api.types.is_datetime64_any_dtype(s):
            info.role = "datetime"
        elif pd.api.types.is_bool_dtype(s):
            info.role, info.kind, info.positive = "binary", "rate", True
            info.polarity = _polarity(tok)
        elif pd.api.types.is_numeric_dtype(s):
            vals = s.dropna()
            is_int = bool((vals % 1 == 0).all())
            if tok & _CODE_TOKENS:
                info.role = "code"
            elif (tok & _ID_TOKENS or _is_sequential_index(s)) and ratio >= 0.9 and is_int:
                info.role = "id"
            elif tok & _YEAR_TOKENS and is_int and vals.between(1900, 2100).all():
                info.role = "year"
            elif tok & _MONTH_TOKENS and is_int and vals.between(1, 12).all():
                info.role = "month"
            elif set(vals.unique()) <= {0, 1} and nun == 2:
                info.role, info.kind, info.positive = "binary", "rate", 1
                info.polarity = _polarity(tok)
            else:
                info.role = "measure"
                info.kind = _measure_kind(tok)
                info.polarity = _polarity(tok)
                info.discrete = is_int and nun <= 12 and n >= 30
        elif _is_texty(s):
            vals = s.dropna().astype(str).str.lower()
            uniq = set(vals.unique())
            avg_len = float(s.dropna().astype(str).str.len().mean()) if len(vals) else 0.0
            if nun == 2 and uniq <= _BOOL_WORDS:
                info.role, info.kind = "binary", "rate"
                info.positive = next((u for u in uniq if u in _POS_BOOL), None)
                info.polarity = _polarity(tok)
                info.positive = next(
                    (orig for orig in s.dropna().unique() if str(orig).lower() == info.positive), info.positive
                )
            elif avg_len > 60:
                info.role = "text"
            elif ratio >= 0.9 and (tok & _ID_TOKENS or ratio >= 0.98):
                info.role = "id"
            elif nun <= MAX_DIM_CARDINALITY and (ratio <= 0.5 or (n < 40 and nun <= 12)):
                info.role = "dimension"
            elif nun > MAX_DIM_CARDINALITY and ratio < 0.9 and n / nun >= 2:
                info.role = "entity"
            else:
                info.role = "id" if ratio >= 0.9 else "text"
        cols[name] = info
    return cols


def _measure_score(df: pd.DataFrame, info: Col) -> float:
    tok = set(tokens(info.name))
    score = sum(w for k, w in _MEASURE_PRIORITY if k in tok)
    if info.role == "binary":
        score += 8 if info.polarity != 0 else -6      # a named outcome flag (churn, attrition, fraud...) is the natural target
    s = pd.to_numeric(df[info.name], errors="coerce").dropna() if info.role != "binary" else pd.Series(dtype=float)
    if len(s) > 3 and abs(s.mean()) > 0:
        score += min(2.0, float(s.std() / abs(s.mean())))
    if info.discrete:
        score -= 1.5
    score += 0.5 * (1 - df[info.name].isna().mean())
    return score


def pick_time(df: pd.DataFrame, cols: dict[str, Col]) -> Optional[dict[str, Any]]:
    candidates = []
    for c in cols.values():
        if c.role != "datetime":
            continue
        s = df[c.name]
        if s.nunique() < 4 or s.notna().mean() < 0.5:
            continue
        tok = set(tokens(c.name))
        score = 2 * len(tok & _DATE_PREFER) - 3 * len(tok & _DATE_AVOID) + float(s.notna().mean()) + min(1.0, s.nunique() / 50)
        candidates.append((score, c.name))
    if candidates:
        return {"kind": "datetime", "col": max(candidates)[1], "label": cols[max(candidates)[1]].label}
    years = []
    for c in cols.values():
        if c.role != "year" or df[c.name].nunique() < 3:
            continue
        tok = set(tokens(c.name))
        score = 2 * len(tok & _YEAR_PREFER) - 4 * len(tok & _YEAR_AVOID)
        if score >= 0:  # attribute-like years (year built, birth year) are not a time axis
            years.append((score, c))
    months = [c for c in cols.values() if c.role == "month"]
    if years:
        y = max(years, key=lambda t: t[0])[1]
        if months:
            return {"kind": "year_month", "col": y.name, "month_col": months[0].name,
                    "label": f"{y.label} / {months[0].label}"}
        return {"kind": "year", "col": y.name, "label": y.label}
    return None


class Dataset:
    """A cleaned table plus its inferred semantic schema."""

    def __init__(self, df: pd.DataFrame, name: str = "Dataset", cols: dict[str, Col] | None = None,
                 time: dict[str, Any] | None | str = "auto", measure_order: list[str] | None = None):
        self.df = df
        self.name = name
        self.cols = cols if cols is not None else infer_columns(df)
        self.time_spec = pick_time(df, self.cols) if isinstance(time, str) else time
        if measure_order is None:
            ms = [c for c in self.cols.values() if c.role in ("measure", "binary")]
            measure_order = [c.name for c in sorted(ms, key=lambda c: -_measure_score(df, c))]
        self.measure_order = measure_order
        self._times: Optional[pd.Series] = None

    # ---- schema views --------------------------------------------------------------
    @property
    def measures(self) -> list[str]:
        return [m for m in self.measure_order if self.cols[m].role == "measure"]

    @property
    def rate_measures(self) -> list[str]:
        return [m for m in self.measure_order if self.cols[m].role == "binary"]

    @property
    def all_measures(self) -> list[str]:
        return list(self.measure_order)

    @property
    def dimensions(self) -> list[str]:
        cats = [c.name for c in self.cols.values() if c.role == "dimension"]
        cats = sorted(cats, key=lambda c: (self.cols[c].nunique < 2, abs(self.cols[c].nunique - 6)))
        flags = [c.name for c in self.cols.values() if c.role == "binary"]
        disc = [c.name for c in self.cols.values() if c.role == "measure" and c.discrete and 2 <= c.nunique <= 12]
        return cats + flags + disc

    @property
    def categorical_dimensions(self) -> list[str]:
        return [c.name for c in self.cols.values() if c.role == "dimension"]

    @property
    def entities(self) -> list[str]:
        return [c.name for c in self.cols.values() if c.role == "entity"]

    @property
    def id_columns(self) -> list[str]:
        return [c.name for c in self.cols.values() if c.role == "id"]

    @property
    def primary(self) -> Optional[str]:
        return self.measure_order[0] if self.measure_order else None

    @property
    def outcome_flags(self) -> list[str]:
        """Binary fields with a known good/bad direction (e.g. churn, attrition, fraud)."""
        return [m for m in self.rate_measures if self.cols[m].polarity != 0]

    def seg_text(self, dim: str, group: str) -> str:
        """Readable segment reference: 'West (Region)' or 'Over Time = Yes'."""
        if self.cols[dim].role in ("binary", "measure"):
            return f"{self.label(dim)} = {group}"
        return f"{group} ({self.label(dim)})"

    def seg_short(self, dim: str, group: str) -> str:
        return f"{self.label(dim)} = {group}" if self.cols[dim].role in ("binary", "measure") else str(group)

    def label(self, col: str) -> str:
        return self.cols[col].label if col in self.cols else pretty(col)

    def kind(self, col: str) -> str:
        c = self.cols.get(col)
        return (c.kind if c and c.kind else "intensity")

    def polarity(self, col: str) -> int:
        c = self.cols.get(col)
        return c.polarity if c else 0

    # ---- data access ---------------------------------------------------------------
    def values(self, col: str) -> pd.Series:
        c = self.cols[col]
        s = self.df[col]
        if c.role == "binary":
            if pd.api.types.is_bool_dtype(s):
                return s.astype(float)
            if pd.api.types.is_numeric_dtype(s):
                return s.astype(float)
            mask = s.notna()
            out = (s.astype(str) == str(c.positive)).astype(float)
            return out.where(mask)
        return pd.to_numeric(s, errors="coerce")

    def dim(self, col: str) -> pd.Series:
        s = self.df[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            nn = s.dropna()
            if len(nn) and (nn % 1 == 0).all():
                out = s.round().astype("Int64").astype(str)
                return out.where(s.notna(), "(missing)")
        out = s.astype("object").where(s.notna(), "(missing)")
        return out.astype(str)

    def times(self) -> Optional[pd.Series]:
        if self.time_spec is None:
            return None
        if self._times is not None:
            return self._times
        spec, df = self.time_spec, self.df
        if spec["kind"] == "datetime":
            t = df[spec["col"]]
        elif spec["kind"] == "year":
            t = pd.to_datetime(df[spec["col"]].astype("Int64").astype(str) + "-01-01", errors="coerce")
        else:
            y = df[spec["col"]].astype("Int64").astype(str)
            m = df[spec["month_col"]].astype("Int64").astype(str).str.zfill(2)
            t = pd.to_datetime(y + "-" + m + "-01", errors="coerce")
        self._times = t
        return t

    def subset(self, mask: pd.Series) -> "Dataset":
        return Dataset(self.df.loc[mask], self.name, self.cols, self.time_spec, self.measure_order)

    @property
    def has_time(self) -> bool:
        t = self.times()
        return t is not None and t.notna().sum() >= 4

    def date_range(self) -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
        t = self.times()
        if t is None or t.notna().sum() == 0:
            return None
        return t.min(), t.max()
