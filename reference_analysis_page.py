import streamlit as st

from config.config import REFERENCE_SHORTEN_BASELINE, REFERENCE_SHORTEN_WITH_EDGE_CASES
from results_analysis.analysis import load_shorten_reference_artifacts
from ui.evaluation_views import render_baseline_edge_comparison, render_results_summary

st.title("Shorten Task Reference Analysis")
st.write("Saved shorten-task results from `reference_artifacts/`.")
st.caption(
    f"{REFERENCE_SHORTEN_BASELINE.name} vs {REFERENCE_SHORTEN_WITH_EDGE_CASES.name}"
)

try:
    baseline_df, edge_df = load_shorten_reference_artifacts()
except FileNotFoundError as exc:
    st.error(f"Reference file not found: {exc}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    render_results_summary(baseline_df, "Baseline (50 emails)")
with col2:
    render_results_summary(edge_df, "With edge cases (60 emails)")

st.subheader("Baseline vs Edge Cases")
render_baseline_edge_comparison(baseline_df, edge_df)
