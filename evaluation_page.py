import json

import pandas as pd
import streamlit as st

from config.config import (
    METRICS,
    MODEL_CONFIG_PATH,
    RESULTS_BASELINE_DIR,
    RESULTS_EDGE_DIR,
    TASKS,
    dataset_paths,
)
from generators.evaluation_pipeline import EvaluationPipeline
from results_analysis.analysis import (
    calculate_failures,
    calculate_perfect_scores,
    get_distribution,
    get_per_task,
    get_recommendation,
    load_all_task_results,
)
from ui.evaluation_views import render_baseline_edge_comparison, render_head_to_head
from utils.jsonl_loader import load_jsonl

TONE_OPTIONS = {
    "Professional": "professional",
    "Friendly": "friendly",
    "Sympathetic": "sympathetic",
}

TASK_LABELS = {
    "shorten": "Shorten",
    "lengthen": "Lengthen",
    "tone": "Tone",
}


st.title("Evaluation Pipeline")
st.write(
    "Each run evaluates **all three tasks** — shorten, lengthen, and tone — "
    "comparing gpt-4o-mini and gpt-4.1 on every email in the selected dataset."
)

model_df = pd.DataFrame({"Model A": ["gpt-4o-mini"], "Model B": ["gpt-4.1"]})
st.dataframe(model_df, hide_index=True)

include_edge_cases = st.checkbox("Include edge cases", value=False)

results_root = "results/with_edge_cases" if include_edge_cases else "results/baseline"
paths = dataset_paths(include_edge_cases)

task_rows = []
for task, path in zip(TASKS, paths):
    task_rows.append(
        {
            "Task": TASK_LABELS[task],
            "Dataset": str(path),
            "Emails": len(load_jsonl(path)),
        }
    )
st.subheader("Tasks included in every run")
st.dataframe(pd.DataFrame(task_rows), hide_index=True)
st.caption(f"Results are saved to `{results_root}/` as one CSV per task.")

with st.expander("Tone task setting"):
    st.write(
        "Shorten and lengthen always use their own prompts. "
        "This option only applies to emails in the tone dataset."
    )
    tone_label = st.selectbox(
        "Target tone for tone-task emails",
        options=list(TONE_OPTIONS.keys()),
        index=0,
    )
    tone_style = TONE_OPTIONS[tone_label]

pipeline = EvaluationPipeline(tasks=TASKS, dataset_paths=paths)

if st.button("Run evaluation for all tasks"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(completed, total, task, email_id):
        progress_bar.progress(completed / total if total else 1.0)
        status_text.text(f"Evaluating {task} email {email_id} ({completed}/{total})")

    saved_paths = pipeline.pipeline(
        include_edge_cases=include_edge_cases,
        tone_style=tone_style,
        progress_callback=update_progress,
    )
    progress_bar.progress(1.0)
    status_text.empty()
    st.success(
        f"Saved results for all tasks to `{results_root}`: "
        f"{', '.join(saved_paths.keys())}"
    )

baseline_df = load_all_task_results(RESULTS_BASELINE_DIR)
edge_df = load_all_task_results(RESULTS_EDGE_DIR)
active_df = edge_df if include_edge_cases and edge_df is not None else baseline_df

if active_df is None:
    st.info("Run an evaluation to see results.")
    st.stop()

st.subheader("Average Scores")
st.bar_chart(active_df.groupby("model")[METRICS].mean())

st.subheader("Performance by Task")
st.dataframe(get_per_task(active_df))

st.subheader("Score Distribution")
st.bar_chart(pd.DataFrame(get_distribution(active_df)).T)

st.subheader("Head-to-Head")
render_head_to_head(active_df, "All tasks")

summary = pd.DataFrame(
    {
        "failures": calculate_failures(active_df),
        "perfect_scores": calculate_perfect_scores(active_df),
    }
)
st.subheader("Failures and Perfect Scores")
st.dataframe(summary)

if baseline_df is not None and edge_df is not None:
    st.subheader("Baseline vs Edge Cases")
    st.write("Compare results from `results/baseline` and `results/with_edge_cases`.")
    for task in TASKS:
        render_baseline_edge_comparison(baseline_df, edge_df, task=task)

recommendation = get_recommendation(active_df, METRICS)
st.subheader("Recommendation")
st.success(f"Use **{recommendation['winner']}** as the primary model")
st.write(f"- {recommendation['winner']}: {recommendation['winner_score']} avg score")
st.write(f"- {recommendation['loser']}: {recommendation['loser_score']} avg score")
st.write(f"- Difference: {recommendation['difference']} points")

MODEL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(MODEL_CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump({"primary_model": recommendation["winner"]}, f)
st.caption(f"Saved primary model to `{MODEL_CONFIG_PATH}`")
