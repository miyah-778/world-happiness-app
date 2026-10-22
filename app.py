import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression

# Configuration
st.set_page_config(page_title="World Happiness Explorer", page_icon="🌍", layout="wide")
st.markdown("""<style>
[data-testid="stSidebar"] {background-color:#1e293b;}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {color:#f1f5f9 !important;}
/* Keep the upload area readable, including on dark themes. */
[data-testid="stFileUploaderDropzone"] {background-color:#f1f5f9; border:1px dashed #94a3b8;}
[data-testid="stFileUploaderDropzone"] * {color:#334155 !important;}
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] button * {
    color:#273244 !important;
    -webkit-text-fill-color:#273244 !important;
    opacity:1 !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background-color:#ffffff !important;
    border:1px solid #94a3b8 !important;
    box-shadow:none !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background-color:#e2e8f0 !important;
    border-color:#64748b !important;
}
[data-testid="stFileUploaderDropzone"] svg {fill:#273244 !important; stroke:#273244 !important;}
/* Readable dropdowns in the main content area. */
[data-baseweb="select"] > div {
    background-color:#f8fafc !important;
    border:1px solid #cbd5e1 !important;
    border-radius:9px !important;
    color:#273244 !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] input,
[data-baseweb="select"] svg {
    color:#273244 !important;
    fill:#273244 !important;
    -webkit-text-fill-color:#273244 !important;
}
.block-container {padding-top:2rem; max-width:1250px;}
[data-testid="stMetric"] {background:#f8fafc; border:1px solid #e2e8f0; padding:18px; border-radius:12px;}
[data-testid="stMetric"] * {color:#273244 !important;}
</style>""", unsafe_allow_html=True)

TARGET = "Ladder score"
EXCLUDE = {"Standard error of ladder score", "upperwhisker", "lowerwhisker"}

st.title("🌍 World Happiness Explorer")
st.caption("Explore socioeconomic indicators and predict national happiness using linear regression.")

# File upload
with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    st.caption("Expected: world-happiness-report-2021.csv · Must contain 'Ladder score' and at least two numeric predictor columns.")
if uploaded_file is None:
    st.info("Upload your World Happiness Report 2021 CSV using the sidebar to get started.")
    st.stop()
try:
    df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

if TARGET not in df.columns:
    st.error("Your CSV must contain a column named 'Ladder score'.")
    st.stop()
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

numeric_cols = df.select_dtypes(include="number").columns.tolist()
features = [c for c in numeric_cols if c != TARGET and c not in EXCLUDE]
if len(features) < 2:
    st.error("Your CSV needs at least two numeric predictor columns in addition to 'Ladder score'.")
    st.stop()

st.subheader("Dataset overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", len(df.columns))
c3.metric("Numeric features", len(features))
c4.metric("Average happiness", f"{df[TARGET].mean():.2f}" if df[TARGET].notna().any() else "N/A")
with st.expander("Preview uploaded data", expanded=True):
    st.dataframe(df.head(5), use_container_width=True)
    st.caption("First 5 rows of the uploaded CSV")

# Choose the column + chart type — filters sit directly above the chart
st.divider()
