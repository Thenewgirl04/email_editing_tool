def calculate_win_rate(df):
    """Simple win rate calculation"""
    metrics = ["faithfulness_rating", "completeness_rating", "relevance_rating"]
    df["total_score"] = df[metrics].sum(axis=1)
    pivot = df.pivot(index="email_id", columns="model", values="total_score").dropna()

    model_a, model_b = pivot.columns
    wins_a = (pivot[model_a] > pivot[model_b]).sum()
    wins_b = (pivot[model_b] > pivot[model_a]).sum()

    return {model_a: wins_a, model_b: wins_b}


def get_distribution(df):
    """Count 1s, 2s, 3s per model"""
    metrics = ["faithfulness_rating", "completeness_rating", "relevance_rating"]

    dist = {}
    for model in df["model"].unique():
        scores = df[df["model"] == model][metrics].values.flatten()
        dist[model] = {"3s": (scores == 3).sum(), "2s": (scores == 2).sum(), "1s": (scores == 1).sum()}

    return dist


def get_per_task(df):
    """Average score per task"""
    metrics = ["faithfulness_rating", "completeness_rating", "relevance_rating"]
    return df.groupby(["task", "model"])[metrics].mean().round(2)


def get_recommendation(df, metrics):
    """Determine which model is better overall"""
    averages = df.groupby("model")[metrics].mean()

    # Calculate overall average for each model
    overall_avg = averages.mean(axis=1)

    winner = overall_avg.idxmax()
    winner_score = overall_avg.max()
    loser = overall_avg.idxmin()
    loser_score = overall_avg.min()

    difference = winner_score - loser_score

    return {
        "winner": winner,
        "winner_score": round(winner_score, 2),
        "loser": loser,
        "loser_score": round(loser_score, 2),
        "difference": round(difference, 2)
    }