import streamlit as st

def upload_file():
    uploaded_file = st.file_uploader("📂 Upload Dataset (CSV)", type=["csv"])
    return uploaded_file
