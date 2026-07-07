import streamlit as st
import pandas as pd
import json
from generators.evaluation_pipeline import EvaluationPipeline
from results_analysis.analysis import get_recommendation



tasks = ["lengthen", "shorten", "tone"]
datasets = ['datasets/shorten.jsonl', 'datasets/lengthen.jsonl', 'datasets/tone.jsonl']

eval = EvaluationPipeline(tasks=tasks, datasets=datasets)

st.title("Evaluation Page")
st.write("Use this pipeline to evaluate which model is good for the email editing task")


data = {
    "Model_A": ["gpt-4o-mini"],
    "Model_B": ["gpt-4.1"],
}

df = pd.DataFrame(data)
st.dataframe(data, hide_index=True)

if st.button("Evaluate"):
    st.write(eval.pipeline())

# Load all three datasets
shorten_df = pd.read_csv('results/shorten_results.csv', encoding='latin-1')
lengthen_df = pd.read_csv('results/lengthen_results.csv', encoding='latin-1')
tone_df = pd.read_csv('results/tone_results.csv', encoding='latin-1')

# Add task column to each
shorten_df['task'] = 'shorten'
lengthen_df['task'] = 'lengthen'
tone_df['task'] = 'tone'

# Combine all
df = pd.concat([shorten_df, lengthen_df, tone_df], ignore_index=True)

metrics = ["faithfulness_rating", "completeness_rating", "relevance_rating"]

# Chart 1: Average scores by model
st.subheader("Average Scores")
averages = df.groupby("model")[metrics].mean()
st.bar_chart(averages)

# Chart 2: Scores by task
st.subheader("Performance by Task")
by_task = df.groupby(["task", "model"])[metrics].mean().reset_index()  # Add reset_index()
st.dataframe(by_task)  # Use dataframe instead of line_chart

# Chart 3: Distribution
st.subheader("Score Distribution")
dist_data = {}
for model in df["model"].unique():
    scores = df[df["model"] == model][metrics].values.flatten()
    dist_data[model] = {"3s": (scores == 3).sum(), "2s": (scores == 2).sum(), "1s": (scores == 1).sum()}
st.bar_chart(pd.DataFrame(dist_data).T)

recommendation = get_recommendation(df, metrics)
st.subheader("Recommendation")
st.success(f"Use **{recommendation['winner']}** as the primary model")
st.write(f"- {recommendation['winner']}: {recommendation['winner_score']} avg score")
st.write(f"- {recommendation['loser']}: {recommendation['loser_score']} avg score")
st.write(f"- Difference: {recommendation['difference']} points")

with open('config/model_config.json', 'w') as f:
    json.dump({'primary_model': recommendation['winner']}, f)

st.success(f"Saved recommendation: {recommendation['winner']}")