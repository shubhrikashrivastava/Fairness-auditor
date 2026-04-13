import streamlit as st
from components.upload import upload_file
from components.charts import plot_distribution
from components.results import show_results
from utils.api import detect_bias, mitigate_bias

st.set_page_config(page_title="Fairness Auditor", layout="wide")

st.title("⚖️ Fairness Auditor")
st.write("Detecting and fixing dataset bias before training AI models")

# Upload
file = upload_file()

if file:
    col1, col2 = st.columns(2)

    # Detect Bias
    with col1:
        st.subheader("🔍 Bias Detection")

        if st.button("Detect Bias"):
            result = detect_bias(file)

            plot_distribution(result["distribution"], "Before SMOTE")

            if result["is_biased"]:
                st.warning("⚠️ Dataset is Biased")
            else:
                st.success("✅ Dataset is Balanced")

    # Fix Bias
    with col2:
        st.subheader("⚖️ Bias Mitigation")

        if st.button("Fix Bias"):
            result = mitigate_bias(file)

            plot_distribution(result["after_distribution"], "After SMOTE")

            show_results(
                result["before_accuracy"],
                result["after_accuracy"]
            )
