import streamlit as st

def show_results(before_acc, after_acc):
    st.subheader("📊 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Before Accuracy",
            value=f"{before_acc:.2f}"
        )

    with col2:
        st.metric(
            label="After Accuracy",
            value=f"{after_acc:.2f}",
            delta=f"{after_acc - before_acc:.2f}"
        )

    # Improvement %
    if before_acc != 0:
        improvement = ((after_acc - before_acc) / before_acc) * 100
        st.markdown(f"📈 **Accuracy Improvement:** {improvement:.2f}%")

    # Interpretation
    if after_acc > before_acc:
        st.success("✅ Model performance improved after bias mitigation")
    elif after_acc == before_acc:
        st.info("ℹ️ Model performance remained the same")
    else:
        st.warning("⚠️ Accuracy decreased, but fairness may still be improved")
