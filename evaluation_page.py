import streamlit as st
import pandas as pd
from generators.evaluation_pipeline import EvaluationPipeline


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




