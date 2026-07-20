from pathlib import Path

import pandas as pd

from config.config import METRICS, REFERENCE_SHORTEN_BASELINE, REFERENCE_SHORTEN_WITH_EDGE_CASES

REFERENCE_COLUMNS = [
    "email_id",
    "data_type",
    "input",
    "model_output",
    "model",
    *METRICS,
    "faithfulness_explanation",
    "completeness_explanation",
    "relevance_explanation",
]


def calculate_averages(df):
    averages = df.groupby("model")[METRICS].mean().round(2)
    averages["Overall"] = averages.mean(axis=1).round(2)
    return averages


def calculate_head_to_head(df):
    working = df.copy()
    working["total_score"] = working[METRICS].sum(axis=1)
    pivot = working.pivot(index="email_id", columns="model", values="total_score").dropna()

    if len(pivot.columns) != 2:
        return None

    model_a, model_b = pivot.columns[0], pivot.columns[1]
    return {
        model_a: int((pivot[model_a] > pivot[model_b]).sum()),
        model_b: int((pivot[model_b] > pivot[model_a]).sum()),
        "ties": int((pivot[model_a] == pivot[model_b]).sum()),
    }


def calculate_failures(df):
    failure_counts = {}
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        failures = model_df[METRICS].eq(1).sum().sum()
        failure_counts[model] = int(failures)
    return failure_counts


def calculate_perfect_scores(df):
    perfect_counts = {}
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        perfect = (
            (model_df["faithfulness_rating"] == 3)
            & (model_df["completeness_rating"] == 3)
            & (model_df["relevance_rating"] == 3)
        ).sum()
        perfect_counts[model] = int(perfect)
    return perfect_counts


def compare_datasets(baseline_df, edge_df):
    baseline_avg = calculate_averages(baseline_df)
    edge_avg = calculate_averages(edge_df)
    difference = (edge_avg - baseline_avg).round(2)
    return {
        "baseline": baseline_avg,
        "with_edge_cases": edge_avg,
        "difference": difference,
    }


def edge_case_impact(baseline_df, edge_df):
    baseline_avg = calculate_averages(baseline_df)
    edge_avg = calculate_averages(edge_df)

    rows = []
    for model in edge_avg.index:
        baseline_score = baseline_avg.loc[model, "Overall"] if model in baseline_avg.index else None
        edge_score = edge_avg.loc[model, "Overall"]
        change = None if baseline_score is None else round(edge_score - baseline_score, 2)
        rows.append(
            {
                "model": model,
                "baseline_overall": baseline_score,
                "with_edge_cases_overall": edge_score,
                "change": change,
            }
        )
    return rows


def calculate_win_rate(df):
    working = df.copy()
    working["total_score"] = working[METRICS].sum(axis=1)
    pivot = working.pivot(index="email_id", columns="model", values="total_score").dropna()

    model_a, model_b = pivot.columns
    wins_a = (pivot[model_a] > pivot[model_b]).sum()
    wins_b = (pivot[model_b] > pivot[model_a]).sum()

    return {model_a: int(wins_a), model_b: int(wins_b)}


def get_distribution(df):
    dist = {}
    for model in df["model"].unique():
        scores = df[df["model"] == model][METRICS].values.flatten()
        dist[model] = {
            "3s": int((scores == 3).sum()),
            "2s": int((scores == 2).sum()),
            "1s": int((scores == 1).sum()),
        }
    return dist


def get_per_task(df):
    return df.groupby(["task", "model"])[METRICS].mean().round(2)


def get_recommendation(df, metrics=None):
    metrics = metrics or METRICS
    averages = df.groupby("model")[metrics].mean()
    overall_avg = averages.mean(axis=1)

    winner = overall_avg.idxmax()
    loser = overall_avg.idxmin()

    return {
        "winner": winner,
        "winner_score": round(float(overall_avg.max()), 2),
        "loser": loser,
        "loser_score": round(float(overall_avg.min()), 2),
        "difference": round(float(overall_avg.max() - overall_avg.min()), 2),
    }


def load_task_results(results_root, task):
    csv_path = results_root / f"{task}_results.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, encoding="latin-1")
    df["task"] = task
    return df


def load_all_task_results(results_root):
    frames = []
    for task in ["shorten", "lengthen", "tone"]:
        task_df = load_task_results(results_root, task)
        if task_df is not None:
            frames.append(task_df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_reference_csv(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, encoding="latin-1")
    available = [column for column in REFERENCE_COLUMNS if column in df.columns]
    return df[available].copy()


def load_shorten_reference_artifacts():
    baseline_df = load_reference_csv(REFERENCE_SHORTEN_BASELINE)
    edge_df = load_reference_csv(REFERENCE_SHORTEN_WITH_EDGE_CASES)
    return baseline_df, edge_df
