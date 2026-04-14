import json
from datetime import datetime

import pandas as pd
import streamlit as st
from components.charts import plot_accuracy_comparison, plot_distribution, plot_donut
from components.results import show_results
from components.theme import apply_dark_theme
from utils.pipeline_runner import preview_csv_columns, run_csv_pipeline, run_demo_pipeline

st.set_page_config(page_title="Fairness Auditor Pro", layout="wide")
apply_dark_theme()

_PIPELINE_KEYS = (
    "before_distribution",
    "after_distribution",
    "before_accuracy",
    "after_accuracy",
    "before_confusion_matrix",
    "after_confusion_matrix",
)


def _pipeline_payload_ok(payload):
    if not isinstance(payload, dict):
        return False
    return all(k in payload for k in _PIPELINE_KEYS)


def _normalize_api_result(result):
    """Ensure we have a dict with full pipeline fields (handles stale or error-shaped JSON)."""
    if not isinstance(result, dict):
        return None
    if result.get("status") == "error":
        return None
    data = result.get("data")
    if _pipeline_payload_ok(data):
        return data
    return None


if "pipeline_data" not in st.session_state:
    st.session_state["pipeline_data"] = None
if "run_history" not in st.session_state:
    st.session_state["run_history"] = []

if st.session_state["pipeline_data"] is not None and not _pipeline_payload_ok(
    st.session_state["pipeline_data"]
):
    st.session_state["pipeline_data"] = None

st.sidebar.title("Controls")
uploaded = st.sidebar.file_uploader(
    "Upload CSV (optional)",
    type=["csv"],
    help="Numeric feature columns + one target column. Pick the target below, then click Refresh Analysis.",
)
if uploaded is None:
    st.session_state.pop("upload_bytes", None)
    st.session_state.pop("upload_name", None)
else:
    st.session_state["upload_bytes"] = uploaded.getvalue()
    st.session_state["upload_name"] = uploaded.name

target_column = None
if st.session_state.get("upload_bytes"):
    cols = preview_csv_columns(st.session_state["upload_bytes"])
    target_column = st.sidebar.selectbox("Target column", cols, help="Column to predict (class labels).")
    st.sidebar.caption("Refresh will run analysis on your uploaded file.")
else:
    st.sidebar.caption("No file: Refresh uses the built-in demo dataset.")
st.sidebar.caption("CSV analysis runs locally (same code as `ml.py`); you do not need Flask for uploads.")

satellite_filter = st.sidebar.selectbox("Filter Segment", ["All", "Class 0", "Class 1"])
time_window = st.sidebar.slider("Time Range (Minutes)", min_value=5, max_value=120, value=(15, 90))
st.sidebar.markdown("---")
st.sidebar.subheader("Quick Stats")
st.sidebar.metric("Total Runs", len(st.session_state["run_history"]))
last_run_time = st.session_state["run_history"][-1]["time"] if st.session_state["run_history"] else "N/A"
st.sidebar.markdown(f"**Last refresh:** {last_run_time}")

head_left, head_mid, head_right = st.columns([2.8, 1.2, 1.2])
with head_left:
    st.markdown('<div class="big-title">Fairness Auditor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="muted">Real-time fairness dashboard for class imbalance detection and bias mitigation.</div>',
        unsafe_allow_html=True,
    )
with head_mid:
    st.markdown('<span class="status-pill">ONLINE</span>', unsafe_allow_html=True)
with head_right:
    if st.button("Refresh Analysis", use_container_width=True):
        try:
            with st.spinner("Running pipeline..."):
                if st.session_state.get("upload_bytes") and target_column:
                    result = run_csv_pipeline(
                        st.session_state["upload_bytes"],
                        target_column,
                    )
                else:
                    result = run_demo_pipeline()
            data_ok = _normalize_api_result(result)
            if data_ok is None:
                raise RuntimeError(
                    "Pipeline returned an incomplete result. "
                    "Use the Streamlit menu: Clear cache, then run Refresh Analysis again."
                )
            st.session_state["pipeline_data"] = data_ok
            st.session_state["last_run_mode"] = result.get("mode", "unknown")
            if st.session_state.get("upload_bytes") and target_column:
                st.session_state["last_run_label"] = (
                    st.session_state.get("upload_name") or "upload.csv"
                )
            else:
                st.session_state["last_run_label"] = "Demo dataset"
            st.session_state["run_history"].append(
                {"time": datetime.now().strftime("%H:%M:%S"), "status": "Success"}
            )
            st.success("Dashboard refreshed.")
        except Exception as exc:
            st.session_state["run_history"].append(
                {"time": datetime.now().strftime("%H:%M:%S"), "status": "Failed"}
            )
            st.error(f"Run failed: {exc}")

if st.session_state["pipeline_data"] is None:
    st.info("No data loaded yet. Click 'Refresh Analysis' to fetch results.")
    st.stop()

data = st.session_state["pipeline_data"]
if not _pipeline_payload_ok(data):
    st.warning("Saved results are missing fields (often from an old session). Click **Refresh Analysis** again.")
    st.session_state["pipeline_data"] = None
    st.stop()

before_dist = data["before_distribution"]
after_dist = data["after_distribution"]

total_samples = int(sum(before_dist.values()))
critical_gap = int(max(before_dist.values()) - min(before_dist.values()))
accuracy_delta = data["after_accuracy"] - data["before_accuracy"]
imbalance_ratio = min(before_dist.values()) / max(before_dist.values())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Samples", total_samples)
k2.metric("Imbalance Gap", critical_gap)
k3.metric("Accuracy Gain", f"{accuracy_delta:.3f}")
k4.metric("Post-SMOTE Accuracy", f"{data['after_accuracy']:.3f}")

_run_src = st.session_state.get("last_run_label", "Unknown")
_run_mode = st.session_state.get("last_run_mode", "")
st.caption(f"Results source: {_run_src}" + (f" (`{_run_mode}`)" if _run_mode else ""))

tab_dashboard, tab_alerts, tab_analytics, tab_report = st.tabs(
    ["Dashboard", "Alerts", "Analytics", "Report"]
)

with tab_dashboard:
    c1, c2 = st.columns(2)
    with c1:
        plot_distribution(before_dist, "Before SMOTE Distribution")
    with c2:
        plot_distribution(after_dist, "After SMOTE Distribution")
    show_results(data["before_accuracy"], data["after_accuracy"])

with tab_alerts:
    st.subheader("Bias Alerts")
    if imbalance_ratio < 0.3:
        st.error("High Risk: Severe class imbalance detected before mitigation.")
    elif imbalance_ratio < 0.6:
        st.warning("Medium Risk: Moderate class imbalance detected before mitigation.")
    else:
        st.success("Low Risk: Dataset appears reasonably balanced.")

    st.markdown(
        """
        <div class="card">
            <h4>Recommendations</h4>
            <ul>
                <li>Track fairness metrics in every training cycle.</li>
                <li>Validate model behavior across minority classes.</li>
                <li>Log run history and monitor shifts over time.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_analytics:
    c1, c2 = st.columns(2)
    with c1:
        plot_donut(after_dist, "Class Mix After Mitigation")
    with c2:
        plot_accuracy_comparison(data["before_accuracy"], data["after_accuracy"])

    st.subheader("Confusion Matrices")
    d1, d2 = st.columns(2)
    with d1:
        st.caption("Before SMOTE")
        st.dataframe(pd.DataFrame(data["before_confusion_matrix"]), use_container_width=True)
    with d2:
        st.caption("After SMOTE")
        st.dataframe(pd.DataFrame(data["after_confusion_matrix"]), use_container_width=True)

with tab_report:
    st.subheader("Run Summary")
    report_payload = {
        "data_source": st.session_state.get("last_run_label", "Unknown"),
        "run_mode": st.session_state.get("last_run_mode", ""),
        "target_column": target_column,
        "segment_filter": satellite_filter,
        "time_window": time_window,
        "before_accuracy": data["before_accuracy"],
        "after_accuracy": data["after_accuracy"],
        "before_distribution": before_dist,
        "after_distribution": after_dist,
    }
    st.json(report_payload)
    st.download_button(
        "Download JSON Report",
        data=json.dumps(report_payload, indent=2),
        file_name="fairness_report.json",
        mime="application/json",
        use_container_width=True,
    )
