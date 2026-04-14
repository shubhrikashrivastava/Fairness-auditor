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

