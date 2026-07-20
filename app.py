import streamlit as st

page_1 = st.Page("editing_tool_page.py", title="AI Editing email tool")
page_2 = st.Page("evaluation_page.py", title="Evaluation Pipeline")
page_3 = st.Page("reference_analysis_page.py", title="Reference Analysis")

pg = st.navigation([page_1, page_2, page_3], position="top")
pg.run()

