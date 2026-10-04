from __future__ import annotations

import json

from decision_engine import build_evidence
from llm import ask_llm


def investigate_dataset(df):
    evidence = build_evidence(df)
    prompt = f"""
You are producing an evidence-grounded executive analytics brief.

VERIFIED EVIDENCE (computed by Python):
{json.dumps(evidence, indent=2, default=str)}

Create a decision-oriented report using exactly these sections:

### Executive Summary
State the dataset scope and the most important verified signals.

### Performance & Patterns
Explain the strongest measurable patterns, concentration, and trends.

### Data Quality & Risk
Discuss completeness, duplicates, and potential anomalies. Never label an anomaly as a confirmed error.

### Potential Drivers
Discuss only the supplied correlation evidence. Explicitly avoid causal claims.

### Recommended Investigations
Give 3-5 concrete next analyses/questions. Recommendations must be grounded in available evidence.

Rules:
- Do not invent business context.
- Do not fabricate KPIs.
- Never infer causality from correlation.
- If evidence is insufficient, say so.
"""
    report = ask_llm(
        prompt,
        max_output_tokens=900,
        system_instruction="You are a rigorous analytics consultant. Separate evidence, interpretation, and recommendations.",
    )
    return report, evidence["profile"], evidence["anomalies"]
