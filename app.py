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
/* Dark text on light select controls, including their selected values. */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color:#f8fafc !important;
    border:1px solid #cbd5e1 !important;
    border-radius:9px !important;
    color:#273244 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    color:#273244 !important;
    fill:#273244 !important;
    -webkit-text-fill-color:#273244 !important;
}
/* Keep the upload area readable, including on dark themes. */
[data-testid="stFileUploaderDropzone"] {background-color:#f1f5f9; border:1px dashed #94a3b8;}
[data-testid="stFileUploaderDropzone"] * {color:#334155 !important;}
/* Streamlit's upload button has nested text styles that may override its parent. */
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

# Choose the column + chart type
st.divider()
st.subheader("Explore happiness")
st.caption("Compare a selected socioeconomic feature with the happiness ladder score.")
with st.sidebar:
    st.header("Chart filters")
    feature = st.selectbox("X-axis feature", features)
    chart_type = st.selectbox("Chart type", ["Scatter", "Line", "Histogram"])

plot_data = df[[feature, TARGET]].dropna().copy()
if plot_data.empty:
    st.warning("No valid observations for this feature and happiness score.")
else:
    if chart_type == "Scatter":
        fig = px.scatter(plot_data, x=feature, y=TARGET, title=f"Happiness vs {feature}", opacity=0.75)
    elif chart_type == "Line":
        fig = px.line(plot_data.sort_values(feature), x=feature, y=TARGET, title=f"Happiness vs {feature}")
    else:
        fig = px.histogram(plot_data, x=feature, y=TARGET, histfunc="avg", nbins=20, title=f"Average happiness by {feature}")
    fig.update_layout(template="plotly_white", height=460, margin=dict(l=20,r=20,t=55,b=20),
                      yaxis_title="Average happiness score" if chart_type == "Histogram" else "Happiness score")
    st.plotly_chart(fig, use_container_width=True)
    if chart_type == "Histogram":
        st.caption("Histogram bins group the selected feature; bar heights show average happiness, not frequency.")
    elif chart_type == "Line":
        st.caption("Observations are sorted by the selected feature. This is not a time-series chart.")

# Linear regression
st.divider()
st.subheader("Predict happiness")
st.caption("Fit a linear regression using two selected numeric features to predict Ladder score.")
a, b = st.columns(2)
with a:
    x1 = st.selectbox("Predictor 1", features, index=0)
with b:
    x2 = st.selectbox("Predictor 2", features, index=1)

if x1 == x2:
    st.warning("Choose two different predictors to fit the regression.")
else:
    train = df[[x1, x2, TARGET]].dropna()
    if len(train) < 4:
        st.warning("At least four complete observations are required to fit the model.")
    else:
        X = train[[x1, x2]]
        y = train[TARGET]
        model = LinearRegression().fit(X, y)
        st.metric("Training R²", f"{model.score(X, y):.3f}")
        st.caption("R² describes fit on the same data used for training; it is not an out-of-sample accuracy estimate.")
        v1col, v2col = st.columns(2)
        with v1col:
            v1 = st.number_input(f"Enter {x1}", value=float(X[x1].median()), format="%.4f")
        with v2col:
            v2 = st.number_input(f"Enter {x2}", value=float(X[x2].median()), format="%.4f")
        if st.button("Predict happiness", type="primary"):
            new_data = pd.DataFrame({x1: [v1], x2: [v2]})
            prediction = model.predict(new_data)[0]
            st.success(f"Predicted happiness ladder score: {prediction:.3f}")
            st.caption("This is a statistical estimate, not a causal conclusion or guaranteed outcome.")
        with st.expander("Model coefficients"):
            st.write(f"Intercept: {model.intercept_:.4f}")
            st.write(f"{x1}: {model.coef_[0]:.4f}")
            st.write(f"{x2}: {model.coef_[1]:.4f}")
            st.caption(f"Model trained on {len(train)} complete rows.")

# Model purpose and costs
st.divider()
st.subheader("Model purpose and risk")
st.write("This model uses 2021 survey-based happiness data, which may contain sampling and response biases and may not reflect current conditions. Therefore, its predictions should serve only as a historical benchmark, as using them to guide public policy or resource allocation could misdirect funding, overlook vulnerable populations, and create social and economic costs.")
