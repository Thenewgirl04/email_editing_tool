import pandas as pd
import streamlit as st

from config.config import METRICS
from results_analysis.analysis import (
    calculate_averages,
    calculate_failures,
    calculate_head_to_head,
    calculate_perfect_scores,
    compare_datasets,
    edge_case_impact,
    get_distribution,
    get_recommendation,
)

TASK_LABELS = {
    "shorten": "Shorten",
    "lengthen": "Lengthen",
    "tone": "Tone",
}


def render_head_to_head(df, label):
    head_to_head = calculate_head_to_head(df)
    if not head_to_head:
        st.info(f"{label}: need two models to compare head-to-head.")
        return

    models = [key for key in head_to_head if key != "ties"]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"{models[0]} wins", head_to_head[models[0]])
    with col2:
        st.metric("Ties", head_to_head["ties"])
    with col3:
        st.metric(f"{models[1]} wins", head_to_head[models[1]])


def render_results_summary(df, title):
    st.subheader(title)
    st.bar_chart(df.groupby("model")[METRICS].mean())
    st.dataframe(calculate_averages(df))

    st.write("Score distribution")
    st.bar_chart(pd.DataFrame(get_distribution(df)).T)

    render_head_to_head(df, title)

    summary = pd.DataFrame(
        {
            "failures": calculate_failures(df),
            "perfect_scores": calculate_perfect_scores(df),
        }
    )
    st.dataframe(summary)

    recommendation = get_recommendation(df)
    st.success(f"Recommended model: **{recommendation['winner']}** ({recommendation['winner_score']} avg)")


def render_baseline_edge_comparison(baseline_df, edge_df, task=None):
    if task:
        st.markdown(f"#### {TASK_LABELS[task]}")
        baseline_df = baseline_df[baseline_df["task"] == task]
        edge_df = edge_df[edge_df["task"] == task]

    if baseline_df.empty or edge_df.empty:
        return

    comparison = compare_datasets(baseline_df, edge_df)
    st.write("Average scores: baseline vs with edge cases")
    st.dataframe(
        pd.concat(
            [
                comparison["baseline"].assign(dataset="baseline"),
                comparison["with_edge_cases"].assign(dataset="with_edge_cases"),
            ]
        )
    )

    st.write("Overall score change when edge cases are included")
    st.dataframe(pd.DataFrame(edge_case_impact(baseline_df, edge_df)))

    if "data_type" in edge_df.columns:
        edge_only = edge_df[edge_df["data_type"] == "edge_case"]
        if not edge_only.empty:
            st.write("Edge-case emails only")
            st.dataframe(calculate_averages(edge_only))
