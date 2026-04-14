import streamlit as st
from components.upload import upload_file
from components.charts import plot_distribution
from components.results import show_results
from utils.api_client import run_pipeline

st.set_page_config(page_title="Fairness Auditor", layout="wide")

# HEADER
st.title("⚖️ Fairness Auditor")
st.write("Detecting and fixing dataset bias before training AI models")

st.divider()

# INFO
st.info("Click the button below to run the fairness pipeline on a prebuilt dataset")

# (Optional Upload - not used in ML yet)
file = upload_file()

st.divider()

# MAIN BUTTON
if st.button("🚀 Run Fairness Pipeline"):
    with st.spinner("Running fairness pipeline..."):
        result = run_pipeline()
        data = result["data"]

    # CREATE COLUMNS
    col1, col2 = st.columns(2)

    # BEFORE SMOTE
    with col1:
        st.subheader("🔴 Before SMOTE (Biased Data)")
        plot_distribution(data["before_distribution"], "Before SMOTE")

    # AFTER SMOTE
    with col2:
        st.subheader("🟢 After SMOTE (Balanced Data)")
        plot_distribution(data["after_distribution"], "After SMOTE")

    st.divider()

    # PERFORMANCE
    st.subheader("📊 Model Performance")
    show_results(data["before_accuracy"], data["after_accuracy"])

    # OPTIONAL: Confusion Matrix
    with st.expander("📉 View Confusion Matrices"):
        st.write("Before SMOTE:")
        st.write(data["before_confusion_matrix"])

        st.write("After SMOTE:")
        st.write(data["after_confusion_matrix"])

st.divider()

# WORKFLOW SECTION
st.subheader("📌 Workflow")
st.markdown("""
1. Load dataset  
2. Create imbalance  
3. Train model (before SMOTE)  
4. Apply SMOTE  
5. Retrain model  
6. Compare results  
""")

# SIDEBAR
st.sidebar.title("About")
st.sidebar.info(
    "This tool detects bias in datasets and uses SMOTE to balance them before training ML models."
)
st.divider()
st.sidebar.success("🚀 Demo Mode: Prebuilt Dataset")
st.subheader("🎯 Why Fairness Matters")
st.markdown("""
- Biased datasets lead to unfair predictions  
- Critical in domains like healthcare and hiring  
- Our system ensures balanced learning using SMOTE  
""")
