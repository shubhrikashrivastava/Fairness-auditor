import matplotlib.pyplot as plt
import streamlit as st

def plot_distribution(distribution, title):
    classes = list(distribution.keys())
    values = list(distribution.values())

    fig, ax = plt.subplots()
    ax.bar(classes, values)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")

    st.pyplot(fig)
# BIAS INSIGHT
st.divider()
st.subheader("🧠 Bias Insight")

before_vals = list(data["before_distribution"].values())
imbalance_ratio = min(before_vals) / max(before_vals)

if imbalance_ratio < 0.3:
    st.error("⚠️ Severe class imbalance detected")
elif imbalance_ratio < 0.6:
    st.warning("⚠️ Moderate imbalance detected")
else:
    st.success("✅ Dataset is fairly balanced")
