from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.detector import ExplainableAnomalyDetector
from src.evaluation import evaluate_against_planted_events
from src.real_data import DEFAULT_TICKERS, download_stock_returns, simple_feature_groups
from src.synthetic_data import feature_groups, make_synthetic_financial_data


st.set_page_config(
    page_title="Explainable Anomaly Detection",
    layout="wide",
)

st.title("Explainable Unsupervised Anomaly Detection")
st.caption("A resume-ready prototype for finding unusual days in high-dimensional financial data.")

with st.sidebar:
    st.header("Demo Settings")
    data_source = st.radio(
        "Data source",
        ["Synthetic demo data", "Real yfinance market data"],
    )
    if data_source == "Synthetic demo data":
        days = st.slider("Number of days", 400, 1200, 900, 50)
        assets = st.slider("Number of assets", 20, 150, 80, 10)
    else:
        ticker_text = st.text_area(
            "Tickers",
            value=", ".join(DEFAULT_TICKERS),
            height=100,
        )
        start_date = st.text_input("Start date", value="2020-01-01")
    contamination = st.slider("Expected anomaly rate", 0.005, 0.08, 0.02, 0.005)
    random_state = st.number_input("Random seed", value=42, step=1)
    run_button = st.button("Run detector", type="primary")

if "result" not in st.session_state or run_button:
    try:
        if data_source == "Synthetic demo data":
            dataset = make_synthetic_financial_data(
                days=days,
                assets=assets,
                random_state=int(random_state),
            )
            model_data = dataset.data
            feature_names = [
                column
                for column in model_data.columns
                if column not in {"planted_anomaly", "planted_type"}
            ]
            groups = feature_groups(feature_names)
            data_note = "Synthetic data with planted answer-key anomalies."
        else:
            tickers = [ticker.strip() for ticker in ticker_text.split(",") if ticker.strip()]
            model_data = download_stock_returns(tickers=tickers, start=start_date, timeout=20)
            feature_names = list(model_data.columns)
            groups = simple_feature_groups(feature_names)
            data_note = "Real daily stock returns downloaded through yfinance."

        detector = ExplainableAnomalyDetector(contamination=contamination)
        result = detector.fit_predict(model_data, groups)
        st.session_state["detector"] = detector
        st.session_state["result"] = result
        st.session_state["data_note"] = data_note
    except Exception as exc:
        st.error(str(exc))
        st.stop()

detector = st.session_state["detector"]
result = st.session_state["result"]
data_note = st.session_state["data_note"]
scored = result.scored_data
metrics = evaluate_against_planted_events(scored)

st.subheader("What this project does")
st.write(
    "The app creates many related financial features, plants a few unusual events, "
    "or downloads real market returns, then asks an unsupervised model to find strange days."
)
st.caption(data_note)

metric_cols = st.columns(5)
metric_cols[0].metric("Days checked", f"{len(scored):,}")
metric_cols[1].metric("Features", f"{len(detector.feature_names):,}")
metric_cols[2].metric("Model flags", int(scored["model_flag"].sum()))
metric_cols[3].metric("Highest severity", f"{scored['severity_percentile'].max():.1%}")
if metrics:
    metric_cols[4].metric("Recall on planted events", metrics.get("recall", 0))
else:
    metric_cols[4].metric("Real-data labels", "not available")

st.info(detector.pca_summary)

st.subheader("Anomaly score over time")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(scored.index, scored["anomaly_score"], color="#264653", linewidth=1.4)
flagged = scored[scored["model_flag"]]
ax.scatter(flagged.index, flagged["anomaly_score"], color="#e76f51", s=35, label="Flagged by model")
if "planted_anomaly" in scored.columns:
    actual = scored[scored["planted_anomaly"]]
    ax.scatter(actual.index, actual["anomaly_score"], facecolors="none", edgecolors="#2a9d8f", s=80, label="Planted anomaly")
ax.set_xlabel("Date")
ax.set_ylabel("Anomaly score")
ax.legend(loc="upper left")
ax.grid(alpha=0.2)
st.pyplot(fig)

left, right = st.columns([1, 1])

with left:
    st.subheader("Top flagged days")
    display_columns = [
        "anomaly_score",
        "severity_percentile",
        "model_flag",
    ]
    if "planted_type" in result.top_anomalies.columns:
        display_columns.append("planted_type")
    st.dataframe(
        result.top_anomalies[display_columns].style.format(
            {
                "anomaly_score": "{:.3f}",
                "severity_percentile": "{:.1%}",
            }
        ),
        use_container_width=True,
    )

with right:
    st.subheader("Plain-English explanation")
    choices = [str(index.date()) for index in result.top_anomalies.index]
    selected_date = st.selectbox("Choose a flagged day", choices)
    selected_timestamp = pd.Timestamp(selected_date)
    row = scored.loc[selected_timestamp]
    st.write(row["plain_explanation"])
    if "planted_type" in scored.columns:
        st.write(f"Planted event type in this demo: **{row['planted_type']}**")
    else:
        st.write("Real data does not include a hidden answer key, so this needs human inspection.")
    st.write(f"Severity percentile: **{row['severity_percentile']:.1%}**")

st.subheader("How to explain this in an interview")
st.write(
    "I built a model that looks for days that are hard to explain using normal market behavior. "
    "Because the data has many features, I first compress noisy dimensions with PCA. "
    "Then Isolation Forest gives each day an anomaly score. "
    "Finally, I explain each flag by showing which assets moved the furthest from their usual range."
)

csv = scored.to_csv().encode("utf-8")
st.download_button(
    "Download scored data",
    data=csv,
    file_name="scored_anomaly_data.csv",
    mime="text/csv",
)
