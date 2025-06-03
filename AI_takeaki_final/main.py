
import streamlit as st
from query import ask_ai

st.set_page_config(page_title="AIたけあき", layout="centered")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://github.com/Takeaki920/AI-Takeaki-2/blob/main/AI_takeaki_final/assets/bg.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    .block-container {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem 2rem 1rem 2rem;
        border-radius: 1.25rem;
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
        max-width: 700px;
        margin: auto;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='text-align: center; margin-top: 30px;'>
        <img src="https://github.com/Takeaki920/AI-Takeaki-2/blob/main/AI_takeaki_final/assets/icon.png" width="120" style="border-radius: 50%;">
        <h1 style='font-size: 2.5rem; margin-top: 10px;'>AIたけあき</h1>
    </div>
    """,
    unsafe_allow_html=True
)

query = st.text_input("質問をどうぞ", placeholder="例：明るい未来のためにどうしたらいい？")

if query:
    with st.spinner("考え中..."):
        response = ask_ai(query)
    st.markdown("### 回答")
    st.markdown(f"> {response}")
