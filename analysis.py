import pandas as pd

def calculate_averages(df):
    """Calculate average scores per model"""
    metrics = ["faithfulness_rating", "completeness_rating", "relevance_rating", "word_count_rating"]
    
    averages = df.groupby("model")[metrics].mean().round(2)
    averages.columns = ["Faithfulness", "Completeness", "Relevance", "Word Count"]
    averages["Overall"] = averages.mean(axis=1).round(2)
    
    return averages

def calculate_head_to_head(df):
    """Calculate head-to-head wins between models"""
    metrics = ["faithfulness_rating", "completeness_rating", "relevance_rating", "word_count_rating"]
    
    df["total_score"] = df[metrics].sum(axis=1)
    pivot = df.pivot(index="email_id", columns="model", values="total_score").dropna()
    
    if len(pivot.columns) != 2:
        return None
    
    model_a, model_b = pivot.columns[0], pivot.columns[1]
    
    return {
        model_a: int((pivot[model_a] > pivot[model_b]).sum()),
        model_b: int((pivot[model_b] > pivot[model_a]).sum()),
        "ties": int((pivot[model_a] == pivot[model_b]).sum())
    }

def calculate_failures(df):
    """Count scores of 1 per model"""
    metrics = ["faithfulness_rating", "completeness_rating", "relevance_rating", "word_count_rating"]
    
    failure_counts = {}
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        failures = model_df[metrics].eq(1).sum().sum()
        failure_counts[model] = int(failures)
    
    return failure_counts

def calculate_perfect_scores(df):
    """Count all-3s per model"""
    perfect_counts = {}
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        perfect = ((model_df["faithfulness_rating"] == 3) & 
                   (model_df["completeness_rating"] == 3) & 
                   (model_df["relevance_rating"] == 3) & 
                   (model_df["word_count_rating"] == 3)).sum()
        perfect_counts[model] = int(perfect)
    
    return perfect_counts

def compare_datasets(original_df, edge_df):
    """Compare original vs edge case averages"""
    original_avg = calculate_averages(original_df)
    edge_avg = calculate_averages(edge_df)
    diff = (edge_avg - original_avg).round(2)
    
    return {
        "original": original_avg,
        "edge_cases": edge_avg,
        "difference": diff
    }

def full_analysis(csv_file, edge_case_file=None):
    """Run complete analysis and return all results"""
    df = pd.read_csv(csv_file)
    
    results = {
        "averages": calculate_averages(df),
        "head_to_head": calculate_head_to_head(df),
        "failures": calculate_failures(df),
        "perfect_scores": calculate_perfect_scores(df)
    }
    
    if edge_case_file:
        edge_df = pd.read_csv(edge_case_file)
        results["comparison"] = compare_datasets(df, edge_df)
    
    return results