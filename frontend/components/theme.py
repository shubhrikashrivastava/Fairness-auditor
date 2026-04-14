import streamlit as st


def apply_dark_theme():
    st.markdown(
        """
        <style>
            /* Base: high-contrast text on dark background */
            .stApp {
                background: radial-gradient(circle at top left, #151a2f, #0c1020 45%, #080b16);
                color: #f0f4ff;
            }
            .main .block-container {
                max-width: 1100px;
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            section.main p,
            section.main li,
            section.main td,
            section.main th,
            section.main strong,
            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] li {
                color: #f0f4ff !important;
            }
            [data-testid="stCaption"] {
                color: #c8d4f0 !important;
            }
            h1, h2, h3, h4 {
                color: #ffffff !important;
            }
            /* Sidebar */
            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #1a223b 0%, #141b31 100%);
                border-right: 1px solid #2a3458;
                color: #f0f4ff;
            }
            section[data-testid="stSidebar"] p,
            section[data-testid="stSidebar"] span,
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
                color: #eef2ff !important;
            }
            /* Widget labels (selectbox, slider, uploader) */
            label[data-testid="stWidgetLabel"] p,
            label[data-testid="stWidgetLabel"] span {
                color: #eef2ff !important;
            }
            /* Metrics */
            div[data-testid="stMetric"] {
                background: rgba(29, 35, 58, 0.92);
                border: 1px solid #3d4a78;
                border-radius: 14px;
                padding: 14px;
            }
            div[data-testid="stMetricLabel"] {
                color: #c8d4f0 !important;
            }
            div[data-testid="stMetricValue"] {
                color: #ffffff !important;
            }
            div[data-testid="stMetricDelta"] {
                color: #a8f5c8 !important;
            }
            .card {
                background: rgba(24, 30, 50, 0.92);
                border: 1px solid #3d4a78;
                border-radius: 16px;
                padding: 18px;
                margin-bottom: 14px;
                color: #f0f4ff;
            }
            .card h4 {
                color: #ffffff !important;
            }
            .card li {
                color: #eef2ff !important;
            }
            .big-title {
                font-size: 2.1rem;
                font-weight: 700;
                margin-bottom: 0.2rem;
                color: #ffffff;
            }
            .muted {
                color: #d0daf5;
                margin-bottom: 0.8rem;
            }
            .status-pill {
                display: inline-block;
                background: rgba(40, 190, 120, 0.2);
                color: #7af0b5;
                border: 1px solid rgba(91, 232, 159, 0.55);
                border-radius: 999px;
                padding: 4px 10px;
                font-size: 0.75rem;
                font-weight: 700;
                margin-bottom: 0.8rem;
            }
            .stButton > button {
                border-radius: 10px;
                border: 1px solid #4a5a8f;
                background: linear-gradient(135deg, #2f3d66, #232d4d);
                color: #ffffff;
                font-weight: 600;
                padding: 0.55rem 1rem;
            }
            .stButton > button:hover {
                border-color: #6f86c4;
                color: #ffffff;
            }
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            .stTabs [data-baseweb="tab"] {
                background: #1a2340;
                border: 1px solid #3d4a78;
                border-radius: 10px;
                color: #eef2ff !important;
                padding: 8px 14px;
            }
            .stTabs [aria-selected="true"] {
                background: #2d3a63 !important;
                color: #ffffff !important;
                border-color: #5c6fa8 !important;
            }
            /* Dataframe / JSON */
            [data-testid="stDataFrame"] {
                color: #f0f4ff;
            }
            .stJson {
                color: #e8ecff !important;
            }
            /* Alerts: keep readable on colored backgrounds */
            div[data-testid="stAlert"] p,
            div[data-testid="stAlert"] div {
                color: inherit;
            }
            /* File uploader: visible on dark background */
            [data-testid="stFileUploader"] {
                background: rgba(30, 38, 62, 0.9);
                border: 1px solid #6b7cc4;
                border-radius: 10px;
                padding: 10px;
            }
            [data-testid="stFileUploader"] section,
            [data-testid="stFileUploader"] span,
            [data-testid="stFileUploader"] small,
            [data-testid="stFileUploader"] label,
            [data-testid="stFileUploader"] p,
            [data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzoneInstructions"] {
                color: #f0f4ff !important;
            }
            [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
                background: rgba(24, 30, 50, 0.75) !important;
                border: 1px dashed #8b9fe0 !important;
                color: #f0f4ff !important;
            }
            [data-testid="stFileUploader"] button {
                color: #ffffff !important;
                background: #3d4d7a !important;
                border: 1px solid #8b9fe0 !important;
            }
            [data-testid="stFileUploader"] button:hover {
                border-color: #b8c5ff !important;
                color: #ffffff !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
