import matplotlib.pyplot as plt
import streamlit as st


def plot_distribution(distribution, title):
    classes = list(distribution.keys())
    values = list(distribution.values())

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#12182d")
    ax.set_facecolor("#12182d")
    ax.bar(classes, values, color="#6c8bff", edgecolor="#9fb1ff", linewidth=1.2)
    ax.set_title(title, color="#e8ecff")
    ax.set_xlabel("Class", color="#c7cfeb")
    ax.set_ylabel("Count", color="#c7cfeb")
    ax.tick_params(colors="#c7cfeb")
    for spine in ax.spines.values():
        spine.set_color("#33406b")

    st.pyplot(fig)


def plot_donut(distribution, title):
    labels = list(distribution.keys())
    sizes = list(distribution.values())
    colors = ["#33d6a6", "#ffb020", "#ff5c7a", "#4f7cff"]

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#12182d")
    ax.set_facecolor("#12182d")
    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors[: len(labels)],
        wedgeprops={"width": 0.45, "edgecolor": "#12182d"},
        textprops={"color": "#d4ddff"},
    )
    ax.set_title(title, color="#e8ecff")
    st.pyplot(fig)


def plot_accuracy_comparison(before_acc, after_acc):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    fig.patch.set_facecolor("#12182d")
    ax.set_facecolor("#12182d")
    labels = ["Before", "After"]
    values = [before_acc, after_acc]
    bars = ax.bar(labels, values, color=["#ff7a59", "#4f7cff"], width=0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy", color="#c7cfeb")
    ax.set_title("Model Accuracy Comparison", color="#e8ecff")
    ax.tick_params(colors="#c7cfeb")
    for spine in ax.spines.values():
        spine.set_color("#33406b")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.2f}",
            ha="center",
            color="#dfe6ff",
            fontsize=10,
        )
    st.pyplot(fig)
