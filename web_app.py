from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis import load_data, profile_dataset
from anomaly import detect_anomalies
from decision_engine import build_evidence
from investigator import investigate_dataset
from llm import provider_name
from query import answer_question
from reporting import build_report
from visuals import auto_charts
from segmentation import segment_dataset


st.set_page_config(page_title="AI Data Detective", page_icon="🔎", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1450px; padding-top: 1.5rem;}
.hero-title {font-size: 2.7rem; font-weight: 750; margin-bottom: 0.1rem;}
.hero-sub {font-size: 1.0rem; opacity: .72; margin-bottom: 1.2rem;}
.small-note {font-size:.82rem; opacity:.7;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">🔎 AI Data Detective</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Decision intelligence for heterogeneous datasets — evidence first, AI second.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Workspace")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    st.divider()
    st.caption("AI layer")
    st.code(provider_name(), language="text")
    st.caption("LLM plans and explains. Python performs the numerical analysis.")
    st.divider()
    st.caption("Privacy")
    st.write("The application computes the dataset profile and analytical evidence locally. The LLM receives structured analytical context rather than the full raw table for ordinary queries.")

if uploaded is None:
    st.info("Upload a dataset to begin a decision-intelligence session.")
    a, b, c = st.columns(3)
    a.markdown("**Executive intelligence**\n\nAutomatic findings, trends, risks and next investigations.")
    b.markdown("**Evidence-grounded AI**\n\nStructured query plans with deterministic Python execution.")
    c.markdown("**Business exploration**\n\nAdaptive charts, anomaly analysis and explainable answers.")
    st.stop()

try:
    df = load_data(uploaded, uploaded.name)
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

if df.empty:
    st.warning("The dataset contains no usable records.")
    st.stop()

profile = profile_dataset(df)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Records", f"{profile['rows']:,}")
m2.metric("Features", profile["columns"])
m3.metric("Missing", f"{profile['missing_cells']:,}")
m4.metric("Duplicates", f"{profile['duplicate_rows']:,}")
m5.metric("Data Health", f"{profile['quality_score']}/100")

st.caption(
    f"{len(profile['numeric_columns'])} numeric · {len(profile['categorical_columns'])} categorical · "
    f"{len(profile['datetime_columns'])} datetime · {provider_name()}"
)

tabs = st.tabs(["Executive Dashboard", "Data Quality", "Explore", "Segments", "AI Copilot", "Report"])

with tabs[0]:
    st.subheader("Executive Dashboard")
    evidence = build_evidence(df)

    if evidence["trends"]:
        t = evidence["trends"][0]
        change = t["period_change_pct"]
        label = "No baseline" if change is None else f"{change:+.1f}%"
        st.metric(f"{t['measure']} period movement", label)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Key Metrics")
        if evidence["metrics"]:
            metric_df = pd.DataFrame(evidence["metrics"][:8])
            st.dataframe(metric_df, use_container_width=True, hide_index=True)
        else:
            st.info("No numeric measures detected.")
    with c2:
        st.markdown("### Potential Drivers")
        if evidence["drivers"]:
            driver_df = pd.DataFrame(evidence["drivers"][:6])
            st.dataframe(driver_df[["feature", "target", "correlation"]], use_container_width=True, hide_index=True)
        else:
            st.info("No suitable numeric driver relationships detected.")

    st.markdown("### Investigation Snapshot")
    anomalies = evidence["anomalies"]
    if anomalies:
        st.warning(f"{sum(v['count'] for v in anomalies.values())} potential anomaly findings across {len(anomalies)} numeric fields.")
    else:
        st.success("No obvious numerical anomalies detected by the available methods.")

    if st.button("Run Executive Investigation", type="primary", use_container_width=True):
        with st.spinner("Building evidence and generating the executive brief…"):
            report, _, _ = investigate_dataset(df)
        st.session_state.executive_report = report

    if st.session_state.get("executive_report"):
        st.markdown(st.session_state.executive_report)

with tabs[1]:
    st.subheader("Data Quality & Risk")
    schema = pd.DataFrame(profile["columns_detail"])
    st.dataframe(schema, use_container_width=True, hide_index=True)

    anomalies = detect_anomalies(df)
    st.markdown("### Potential Anomalies")
    if not anomalies:
        st.success("No obvious numerical anomalies detected.")
    else:
        for col, info in anomalies.items():
            st.warning(f"{col}: {info['count']} potential anomaly rows — {info['method']}")
            st.dataframe(info["rows"], use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Adaptive Exploratory Analysis")
    charts = auto_charts(df)
    if charts:
        for fig in charts:
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    else:
        st.info("No suitable visualizations were detected for this schema.")

with tabs[3]:
    st.subheader("🧩 Data-Driven Segmentation")
    segmentation = segment_dataset(df)
    if not segmentation.get("available"):
        st.info(segmentation.get("reason", "Segmentation is not available for this dataset."))
    else:
        st.caption("K-means groups records using standardized numeric features. Treat these clusters as statistical segments, not causal or business personas.")
        seg_df = pd.DataFrame(segmentation["profiles"])
        st.metric("Segments detected", segmentation["clusters"])
        st.dataframe(seg_df, use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("🤖 AI Analyst Copilot")
    st.caption("Ask business questions. The model creates a constrained analytical plan; Python validates and executes it.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("trace"):
                with st.expander("Evidence & analytical trace"):
                    st.json(item["trace"])

    q = st.chat_input("Ask about performance, risks, drivers, trends or the dataset…")
    if q:
        st.session_state.chat_history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Planning → validating → calculating → explaining…"):
                try:
                    answer, plan, exact = answer_question(df, q)
                    st.markdown(answer)
                    trace = {"query_plan": plan.model_dump(), "verified_result": exact}
                    with st.expander("Evidence & analytical trace"):
                        st.json(trace)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer, "trace": trace})
                except Exception as exc:
                    message = f"Analysis could not be completed safely: {exc}"
                    st.error(message)
                    st.session_state.chat_history.append({"role": "assistant", "content": message})

with tabs[5]:
    st.subheader("📋 Decision Report")
    if st.session_state.get("executive_report") is None:
        if st.button("Generate Report", use_container_width=True):
            with st.spinner("Generating report…"):
                report, _, _ = investigate_dataset(df)
            st.session_state.executive_report = report

    report = st.session_state.get("executive_report")
    if report:
        st.markdown(report)
        md = build_report(uploaded.name, df, report)
        st.download_button("Download Markdown Report", md, file_name="ai_data_detective_report.md", mime="text/markdown", use_container_width=True)
    else:
        st.info("Generate the executive investigation to create the report.")
