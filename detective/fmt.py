"""Formatting helpers shared by the analytics engine, UI and reports."""
from __future__ import annotations

import math
import re
from typing import Any, Iterable

import numpy as np

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def tokens(name: Any) -> list[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    s = re.sub(r"[^A-Za-z0-9]+", " ", s).lower()
    return [t for t in s.split() if t]


def pretty(name: Any) -> str:
    parts = tokens(name)
    if not parts:
        return str(name)
    small = {"id", "sku", "usd", "gdp", "roi", "kpi", "ctr", "cpc", "vip"}
    return " ".join(p.upper() if p in small else p.capitalize() for p in parts)


def num(x: Any, decimals: int | None = None) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    x = float(x)
    a = abs(x)
    if decimals is not None:
        return f"{x:,.{decimals}f}"
    if a >= 1e9:
        return f"{x / 1e9:.2f}B"
    if a >= 1e6:
        return f"{x / 1e6:.2f}M"
    if a >= 1000:
        return f"{x:,.0f}"
    if a >= 100:
        return f"{x:,.1f}"
    if a >= 1:
        return f"{x:,.2f}"
    if a == 0:
        return "0"
    return f"{x:.3g}"


def pct(x: Any, signed: bool = False, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    s = f"{float(x):+.{decimals}f}%" if signed else f"{float(x):.{decimals}f}%"
    return s


def value(kind: str | None, x: Any) -> str:
    """Format a measure value according to its kind (rates are shown as percentages)."""
    if kind == "rate":
        return pct(None if x is None else float(x) * 100)
    return num(x)


def month_name(i: int) -> str:
    return _MONTHS[int(i) - 1]


def weekday_name(i: int) -> str:
    return _WEEKDAYS[int(i)]


def period_label(ts, freq: str) -> str:
    if freq == "M":
        return ts.strftime("%b %Y")
    if freq == "Q":
        return f"Q{(ts.month - 1) // 3 + 1} {ts.year}"
    if freq == "Y":
        return str(ts.year)
    if freq == "W":
        return "Week of " + ts.strftime("%d %b %Y")
    return ts.strftime("%d %b %Y")


def unit_word(freq: str) -> str:
    return {"D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}.get(freq, "period")


def join_and(items: Iterable[str]) -> str:
    items = [str(i) for i in items]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


_NUM_TOKEN = re.compile(r"(?<![A-Za-z_])[-+]?\d[\d,]*\.?\d*\s?(?:[KMBkmb]\b|%)?")


def extract_numbers(text: str) -> list[float]:
    """Pull every numeric literal out of free text (handles 1,234 / 1.2M / 12%)."""
    out: list[float] = []
    for m in _NUM_TOKEN.finditer(text or ""):
        raw = m.group(0).strip()
        mult = 1.0
        if raw and raw[-1] in "KkMmBb":
            mult = {"k": 1e3, "m": 1e6, "b": 1e9}[raw[-1].lower()]
            raw = raw[:-1]
        raw = raw.rstrip("%").replace(",", "").strip()
        try:
            out.append(float(raw) * mult)
        except ValueError:
            continue
    return out
