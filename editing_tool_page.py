import streamlit as st
from generators.generate_response import GenerateEmail
from utils.selected_option import option_to_action

# --- CONFIG ---
email_generator = GenerateEmail("gpt-4o-mini")

# --- UI HEADER ---
st.title("📧 AI Email Editing Tool")
st.write("Input your draft email and use AI to refine it")

option = st.selectbox(
    "What would you like to do? ",
    ("Shorten an email", "Lengthen an email", "Change the tone of an email"),
    index=None,
    placeholder="Select what you would like to do..."
)

if option == "Change the tone of an email":
    option = st.selectbox(
        "What tone would you like to change it to? ",
        ("Friendly", "Sympathetic", "Professional"),
        index=None,
        placeholder="Select what you would like to do..."
    )


selected_option = option_to_action(option)

email_text = st.text_area(
    "Email Content",
    height=250,
    placeholder="Enter your email here..."
)


if st.button("Generate"):
    print(option)
    st.write(email_generator.generate(action=selected_option, selected_text=email_text))

