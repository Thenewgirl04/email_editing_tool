import streamlit as st
import json
import requests
import pandas as pd
import altair as alt
import os
from generate import GenerateEmail



# --- CONFIG ---
st.set_page_config(page_title="AI Email Editor", page_icon="📧", layout="wide")
if "generated_email" not in st.session_state:
    st.session_state.generated_email = ""

if "current_page" not in st.session_state:
    st.session_state.current_page = "Generate Email"

# --- HELPER FUNCTION ---
def calculate_word_count(text):
    """Calculate word count from text"""
    return len(text.split())

def calculate_word_count_change(original_text, edited_text):
    """Calculate word count change percentage and direction"""
    original_count = calculate_word_count(original_text)
    edited_count = calculate_word_count(edited_text)

    if original_count == 0:
        return 0, "neutral", original_count, edited_count

    change_percent = round(((edited_count - original_count) / original_count) * 100)

    if change_percent > 0:
        direction = "expansion"
    elif change_percent < 0:
        direction = "reduction"
    else:
        direction = "neutral"

    return change_percent, direction, original_count, edited_count

st.sidebar.title("Page Naviagtion")
page = st.sidebar.radio(
    "Select a page:",
    ["Generate Email", "View Responses", "Analysis"],
    key="page_selector"
)
st.session_state.current_page = page

# --- UI HEADER ---
if st.session_state.current_page == "Generate Email":
    st.title("📧 AI Email Editing Tool")
    st.write("Select a task to evaluate.")

    option = st.selectbox(
        "What task would you like to evaluate ",
        ("Shorten an email", "Lengthen an email", "Change the tone of an email"),
        index=None,
        placeholder="Select what you would like to do..."
    )

    task_label_to_key = {
        "Shorten an email": "shorten",
        "Lengthen an email": "lengthen",
        "Change the tone of an email": "tone",
    }
    task_dataset_map = {
        "shorten": "dataset_copy/shorten.jsonl",
        "lengthen": "dataset_copy/lengthen.jsonl",
        "tone": "dataset_copy/tone.jsonl",
    }
    tone_label_to_action = {
        "Professional": "professional",
        "Friendly": "friendly",
        "Sympathetic": "sympathetic",
    }
    selected_task = task_label_to_key.get(option)

    # selected_model = st.selectbox(
        # "What model would you like to use? ",
        # ("gpt-4o-mini", "gpt-4.1"),
        # index=None,
        # placeholder="Select model...)"

    # if option is None:
    #     st.info("Pick an option to load the dataset.")

    emails = []


    def load_json(file):
        """Load JSONL with resilience to malformed lines (extra data on a line)."""
        data = []
        bad_lines = 0
        decoder = json.JSONDecoder()
        with open(file, "r", encoding="utf-8") as s:
            for idx, line in enumerate(s, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    # Try standard load first
                    data.append(json.loads(stripped))
                except json.JSONDecodeError:
                    # Try to decode only the first JSON object on the line
                    try:
                        obj, end = decoder.raw_decode(stripped)
                        # Accept only if the remaining content is whitespace
                        if stripped[end:].strip():
                            bad_lines += 1
                        else:
                            data.append(obj)
                    except json.JSONDecodeError:
                        bad_lines += 1
        if bad_lines:
            st.warning(f"Skipped {bad_lines} malformed line(s) in {file}.")
        return data


    tone_option = None
    if option == "Change the tone of an email":
        tone_option = st.selectbox(
            "What tone would you prefer? ",
            ("Professional", "Friendly", "Sympathetic"),
            index=None,
            placeholder="Select tone..."
        )


    # --- BASELINE EVALUATION BUTTON ---
    include_edge_cases = st.checkbox("Include Edge Cases", value=False)

    if st.button("Run Evaluation"):
        if tone_option is None:
            st.error("Please select a tone before running evaluation.")
            st.stop()

        # 1. Load all task datasets
        task_emails = {}
        for task, dataset_path in task_dataset_map.items():
            if not os.path.exists(dataset_path):
                st.error(f"Dataset not found for task '{task}': {dataset_path}")
                st.stop()
            records = load_json(dataset_path)
            if not records:
                st.warning(f"No emails found for task '{task}' in {dataset_path}.")
                st.stop()
            task_emails[task] = records

        # 2. Create fresh combined CSV with specified headers
        baseline_csv = "evaluation_all_tasks_results.csv"
        headers = [
            "email_id",
            "data_type",
            "task",
            "input",
            "model_output",
            "model",
            "faithfulness_rating",
            "faithfulness_explanation",
            "completeness_rating",
            "completeness_explanation",
            "relevance_rating",
            "relevance_explanation",
            "word_count_rating",
            "word_count_change_percent",
            "word_count_explanation",
            "original_word_count",
            "edited_word_count",
        ]
        pd.DataFrame(columns=headers).to_csv(baseline_csv, index=False)

        # 3. Set models
        models = ["gpt-4o-mini", "gpt-4.1"]

        # 4. Set evaluator
        evaluator = GenerateEmail(model="gpt-4.1")

        total_iterations = len(models) * sum(len(rows) for rows in task_emails.values())
        progress_bar = st.progress(0)
        status_text = st.empty()
        iteration = 0

        # 5. Loop through each task, model, and email
        for task, task_rows in task_emails.items():
            for model in models:
                generator = GenerateEmail(model=model)

                for email in task_rows:
                    email_content = email["content"]
                    generation_action = task
                    if task == "tone":
                        generation_action = tone_label_to_action[tone_option]

                    result = generator.generate(generation_action, selected_text=email_content)
                    (
                        word_count_change_percent,
                        word_count_change_direction,
                        original_word_count,
                        edited_word_count,
                    ) = calculate_word_count_change(email_content, result)

                    judge_result = evaluator.evaluate(
                        selected_text=email_content,
                        model_response=result,
                        original_word_count=original_word_count,
                        edited_word_count=edited_word_count,
                        word_count_change_percent=word_count_change_percent,
                        word_count_change_direction=word_count_change_direction,
                    )

                    # Mark edge cases for IDs > 50 when include_edge_cases is checked
                    data_type = (
                        "edge_case"
                        if task == "shorten" and include_edge_cases and email["id"] > 50
                        else "baseline"
                    )

                    results_data = {
                        "email_id": email["id"],
                        "data_type": data_type,
                        "task": task,
                        "input": email_content,
                        "model_output": result,
                        "model": model,
                        "faithfulness_rating": judge_result["faithfulness_judge"]["rating"],
                        "faithfulness_explanation": judge_result["faithfulness_judge"]["explanation"],
                        "completeness_rating": judge_result["completeness_judge"]["rating"],
                        "completeness_explanation": judge_result["completeness_judge"]["explanation"],
                        "relevance_rating": judge_result["relevance_judge"]["rating"],
                        "relevance_explanation": judge_result["relevance_judge"]["explanation"],
                        "word_count_rating": judge_result["word_count_judge"]["rating"],
                        "word_count_change_percent": judge_result["word_count_judge"].get(
                            "word_count_change_percent"
                        ),
                        "word_count_explanation": judge_result["word_count_judge"]["explanation"],
                        "original_word_count": original_word_count,
                        "edited_word_count": edited_word_count,
                    }

                    df = pd.DataFrame([results_data])
                    df.to_csv(baseline_csv, mode="a", header=False, index=False)

                    iteration += 1
                    progress = iteration / total_iterations
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Processed {iteration} of {total_iterations} evaluations "
                        f"(task={task}, model={model}, email_id={email['id']})"
                    )

        # 6. After all loops complete: show quick averages for this run
        df = pd.read_csv(baseline_csv)
        metric_cols = [
            "faithfulness_rating",
            "completeness_rating",
            "relevance_rating",
            "word_count_rating",
        ]
        averages = df.groupby("model")[metric_cols].mean().round(2)
        averages["Overall"] = averages.mean(axis=1).round(2)

        st.subheader("Run Averages (this file)")
        st.dataframe(averages)

        st.success(
            "Baseline evaluation complete. "
            "You can view aggregate analysis on the 'Analysis' page."
        )

elif st.session_state.current_page == "View Responses":
    st.title("📊 Email Evaluation Reports")
    st.write("View and analyze generated email evaluation results.")

    task_option = st.selectbox(
        "Select Task",
        ("Shorten an email", "Lengthen an email", "Change the tone of an email"),
        index=0
    )

    task_label_to_key = {
        "Shorten an email": "shorten",
        "Lengthen an email": "lengthen",
        "Change the tone of an email": "tone",
    }
    selected_task = task_label_to_key[task_option]
    csv_file = "evaluation_all_tasks_results.csv"

    if not os.path.exists(csv_file):
        st.warning(f"No results found for {task_option}. Run evaluations first.")
        st.stop()

    df = pd.read_csv(csv_file)

    if "task" not in df.columns:
        st.warning("Results file is missing the 'task' column. Re-run evaluation.")
        st.stop()

    df = df[df["task"] == selected_task]

    if df.empty:
        st.warning("No results found. Run evaluations first.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        models = df["model"].unique()
        selected_model = st.selectbox("Select Model", options=models)

    with col2:
        # Filter email IDs based on selected model
        filtered_ids = df[df["model"] == selected_model]["email_id"].unique()
        email_ids = sorted(filtered_ids)
        selected_id = st.selectbox("Select Email ID", options=email_ids)

    filtered = df[(df["email_id"] == selected_id) & (df["model"] == selected_model)]

    if filtered.empty:
        st.warning(f"No results for Email {selected_id} with {selected_model}")
        st.stop()

    row = filtered.iloc[0]

    st.markdown("### 📥 Original Email")
    st.text_area("Input", value=row["input"], height=200, disabled=True)

    st.markdown("### 📤 Edited Email")
    st.text_area("Output", value=row["model_output"], height=200, disabled=True)

    st.markdown("### 📊 Evaluation Scores")

    score_cols = st.columns(4)

    metrics = [
        ("Faithfulness", "faithfulness_rating", "faithfulness_explanation"),
        ("Completeness", "completeness_rating", "completeness_explanation"),
        ("Relevance", "relevance_rating", "relevance_explanation"),
        ("Word Count", "word_count_rating", "word_count_explanation")
    ]

    for col, (name, rating_col, exp_col) in zip(score_cols, metrics):
        with col:
            st.metric(name, int(row[rating_col]))

    st.markdown("### 📏 Word Count Details")
    wc_cols = st.columns(3)
    with wc_cols[0]:
        st.metric("Original", int(row["original_word_count"]))
    with wc_cols[1]:
        st.metric("Edited", int(row["edited_word_count"]))
    with wc_cols[2]:
        st.metric("Change", f"{row['word_count_change_percent']}%")

    st.markdown("### 💬 Judge Explanations")

    for name, rating_col, exp_col in metrics:
        with st.expander(f"{name} Explanation (Score: {int(row[rating_col])})"):
            st.write(row[exp_col])

elif st.session_state.current_page == "Analysis":
    st.title("📈 Baseline Analysis")
    st.write("Compare models across tasks using aggregate evaluation metrics.")

    task_option = st.selectbox(
        "Select Task to Analyze",
        ("Shorten an email", "Lengthen an email", "Change the tone of an email"),
        index=0
    )
    task_label_to_key = {
        "Shorten an email": "shorten",
        "Lengthen an email": "lengthen",
        "Change the tone of an email": "tone",
    }
    selected_task = task_label_to_key[task_option]

    metric_cols = [
        "faithfulness_rating",
        "completeness_rating",
        "relevance_rating",
        "word_count_rating",
    ]

    def render_section(title, df_section):
        averages = df_section.groupby("model")[metric_cols].mean().round(2)
        averages["Overall"] = averages.mean(axis=1).round(2)

        st.subheader(title)
        st.dataframe(averages)

        chart_df = (
            averages[metric_cols]
            .transpose()
            .rename_axis("metric")
            .reset_index()
            .melt(id_vars="metric", var_name="model", value_name="score")
        )

        bar_chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("metric:N", title="Metric"),
                y=alt.Y("score:Q", title="Average Score"),
                color=alt.Color("model:N", title="Model", scale=alt.Scale(scheme="tableau10")),
                xOffset="model:N",
                tooltip=["metric", "model", "score"],
            )
            .properties(height=320)
        )
        st.altair_chart(bar_chart, use_container_width=True)

        # Head-to-head comparison
        if df_section["model"].nunique() >= 2:
            df_section["total_score"] = (
                df_section["faithfulness_rating"]
                + df_section["completeness_rating"]
                + df_section["relevance_rating"]
                + df_section["word_count_rating"]
            )
            pivot = df_section.pivot_table(index="email_id", columns="model", values="total_score")
            models = list(pivot.columns)
            if len(models) >= 2:
                model_a, model_b = models[0], models[1]
                wins_a = (pivot[model_a] > pivot[model_b]).sum()
                wins_b = (pivot[model_b] > pivot[model_a]).sum()
                ties = (pivot[model_a] == pivot[model_b]).sum()

                st.markdown("**Head-to-Head**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{model_a} Wins", int(wins_a))
                with col2:
                    st.metric("Ties", int(ties))
                with col3:
                    st.metric(f"{model_b} Wins", int(wins_b))
            else:
                st.info("Need at least two models to compute a head-to-head comparison.")
        else:
            st.info("Only one model found; head-to-head comparison is not available.")

        return averages

    combined_file = "evaluation_all_tasks_results.csv"
    if os.path.exists(combined_file):
        combined_df = pd.read_csv(combined_file)
        if "task" not in combined_df.columns:
            st.warning("Combined results file is missing the 'task' column. Re-run evaluation.")
            st.stop()
        task_df = combined_df[combined_df["task"] == selected_task]
        if task_df.empty:
            st.warning(f"No results found for {task_option}. Run evaluations first.")
            st.stop()
        render_section("Average Scores", task_df)
        st.stop()

    # Non-shorten tasks: keep a simple view
    if task_option != "Shorten an email":
        task_to_file = {
            "Lengthen an email": "evaluation_lengthen_results.csv",
            "Change the tone of an email": "evaluation_tone_results.csv",
        }
        csv_file = task_to_file.get(task_option)
        if not os.path.exists(csv_file):
            st.warning(f"No results found for {task_option}. Run evaluations first.")
            st.stop()
        df = pd.read_csv(csv_file)
        if df.empty:
            st.warning("No results found for this task. Run evaluations first.")
            st.stop()
        render_section("Average Scores", df)
        st.stop()

    # Shorten-specific analysis with edge cases
    baseline_file = "evaluation_shorten_results.csv"
    full_file = "evaluation_shorten_with_edgecases_results.csv"

    if not os.path.exists(baseline_file):
        st.warning("No baseline results found. Run baseline without edge cases first.")
        st.stop()

    baseline_df = pd.read_csv(baseline_file)
    if baseline_df.empty:
        st.warning("Baseline file is empty. Run baseline without edge cases first.")
        st.stop()

    # Section 1: Baseline Results (50 emails)
    baseline_avg = render_section("Baseline Results (50 emails)", baseline_df)

    # Section 2: Full Dataset Results (60 emails - Baseline + Edge Cases)
    if not os.path.exists(full_file):
        st.info("Run baseline with 'Include Edge Cases' checked to see full dataset results.")
        st.stop()

    full_df = pd.read_csv(full_file)
    if full_df.empty:
        st.warning("Full dataset file is empty. Re-run with edge cases.")
        st.stop()

    full_avg = render_section("Full Dataset Results (60 emails - Baseline + Edge Cases)", full_df)

    # Failure count (any metric == 0) and perfect scores (all metrics == 3) per model
    def failure_and_perfect_counts(df_src, label):
        failures = (
            df_src.groupby("model")[metric_cols]
            .apply(lambda x: (x.eq(0).any(axis=1)).sum())
            .rename("failure_count")
        )
        perfects = (
            df_src.groupby("model")[metric_cols]
            .apply(lambda x: (x.eq(3).all(axis=1)).sum())
            .rename("perfect_scores")
        )
        summary = pd.concat([failures, perfects], axis=1)
        st.markdown(f"**{label} – Failures & Perfect Scores**")
        st.dataframe(summary)

    failure_and_perfect_counts(full_df, "Full Dataset")

    # Section 3: Impact of Edge Cases
    st.subheader("Impact of Edge Cases")

    # (1) How Edge Cases Affected Model Performance
    impact_rows = []
    for model in full_avg.index:
        base_val = baseline_avg.loc[model, "Overall"] if model in baseline_avg.index else None
        full_val = full_avg.loc[model, "Overall"]
        change = None if base_val is None else round(full_val - base_val, 2)
        impact_rows.append(
            {
                "model": model,
                "baseline_overall": base_val,
                "full_overall": full_val,
                "change_full_minus_baseline": change,
            }
        )
    impact_df = pd.DataFrame(impact_rows)
    st.markdown("**How Edge Cases Affected Model Performance**")
    st.dataframe(impact_df)

    # (2) Edge Cases Only Performance
    if "data_type" in full_df.columns:
        edge_only = full_df[full_df["data_type"] == "edge_case"]
        st.markdown("**Edge Cases Only Performance (10 emails)**")
        if edge_only.empty:
            st.info("No edge case rows found. Re-run with 'Include Edge Cases' checked.")
            edge_avg = pd.DataFrame()
        else:
            edge_avg = edge_only.groupby("model")[metric_cols].mean().round(2)
            edge_avg["Overall"] = edge_avg.mean(axis=1).round(2)
            st.dataframe(edge_avg)
    else:
        st.info("Edge case file missing data_type column. Re-run with 'Include Edge Cases' checked.")
        edge_only = pd.DataFrame()
        edge_avg = pd.DataFrame()

    # (3) Bar chart: baseline vs full per model
    st.markdown("**Side-by-Side: Baseline vs Full (Overall by Model)**")
    combined = []
    for model in full_avg.index:
        if model in baseline_avg.index:
            combined.append({"model": model, "dataset": "Baseline (50)", "overall": baseline_avg.loc[model, "Overall"]})
        combined.append({"model": model, "dataset": "Full (60)", "overall": full_avg.loc[model, "Overall"]})
    combined_df = pd.DataFrame(combined)
    if not combined_df.empty:
        chart_df2 = combined_df
        chart = (
            alt.Chart(chart_df2)
            .mark_bar()
            .encode(
                x=alt.X("model:N", title="Model"),
                y=alt.Y("overall:Q", title="Overall Score"),
                color=alt.Color("dataset:N", title="Dataset", scale=alt.Scale(scheme="dark2")),
                xOffset="dataset:N",
                tooltip=["model", "dataset", "overall"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

    # Section 4: Final Recommendation
    st.subheader("Final Recommendation")
    recommendation_notes = []

    # Overall on full dataset
    full_overall_sorted = full_avg["Overall"].sort_values(ascending=False)
    if len(full_overall_sorted) >= 2 and full_overall_sorted.iloc[0] == full_overall_sorted.iloc[1]:
        recommendation_notes.append("Overall scores on full dataset are tied.")
        best_model_full = full_overall_sorted.index[0]
    else:
        best_model_full = full_overall_sorted.index[0]

    # Head-to-head on full dataset
    head_winner = None
    if full_df["model"].nunique() >= 2:
        full_df["total_score"] = (
            full_df["faithfulness_rating"]
            + full_df["completeness_rating"]
            + full_df["relevance_rating"]
            + full_df["word_count_rating"]
        )
        pivot_full = full_df.pivot_table(index="email_id", columns="model", values="total_score")
        models_full = list(pivot_full.columns)
        if len(models_full) >= 2:
            ma, mb = models_full[0], models_full[1]
            wins_ma = (pivot_full[ma] > pivot_full[mb]).sum()
            wins_mb = (pivot_full[mb] > pivot_full[ma]).sum()
            if wins_ma > wins_mb:
                head_winner = ma
            elif wins_mb > wins_ma:
                head_winner = mb
            recommendation_notes.append(f"Head-to-head wins on full dataset: {ma} {wins_ma} vs {mb} {wins_mb}.")

    # Drop on edge cases (smaller drop is better)
    drop_winner = None
    drops = {}
    for model in full_avg.index:
        base_val = baseline_avg.loc[model, "Overall"] if model in baseline_avg.index else None
        full_val = full_avg.loc[model, "Overall"]
        if base_val is not None:
            drops[model] = round(full_val - base_val, 2)  # negative is a drop
    if drops:
        drop_sorted = sorted(drops.items(), key=lambda x: x[1], reverse=True)
        drop_winner = drop_sorted[0][0]
        recommendation_notes.append(f"Edge-case impact (Overall change): {drops}")

    # Choose recommendation prioritizing full Overall; note other signals
    recommendation = best_model_full
    if head_winner and head_winner != recommendation:
        recommendation_notes.append(f"Head-to-head favors {head_winner}.")
    if drop_winner and drop_winner != recommendation:
        recommendation_notes.append(f"Smaller drop favors {drop_winner}.")

    st.success(f"Based on evaluation of 60 emails including edge cases, {recommendation} is recommended for the email shortening task.")
    if recommendation_notes:
        st.caption(" | ".join(recommendation_notes))