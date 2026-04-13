import streamlit as st

def show_results(before_acc, after_acc):
    st.subheader("📊 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Before Accuracy", before_acc)

    with col2:
        st.metric("After Accuracy", after_acc)

    if after_acc >= before_acc:
        st.success("✅ Fairness Improved")
    else:
        st.warning("⚠️ No Significant Improvement")
