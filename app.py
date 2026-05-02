import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Fraud Dashboard", layout="wide")

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center; color:#38bdf8;'>💳 Fraud Detection Dashboard</h1>
<p style='text-align:center; color:gray;'>AI-Based Transaction Monitoring</p>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")

threshold = st.sidebar.slider("Fraud Threshold", 0.1, 0.9, 0.5)
filter_option = st.sidebar.selectbox("Filter", ["All", "Fraud Only", "Safe Only"])

# ---------------- FILE ----------------
file = st.file_uploader("Upload CSV", type=["csv"])

if file:

    df = pd.read_csv(file)

    # ---------------- PREP ----------------
    if "Class" in df.columns:
        X = df.drop("Class", axis=1)
    else:
        X = df.copy()

    probs = model.predict_proba(X)[:, 1]
    preds = (probs > threshold).astype(int)

    df["Fraud_Probability"] = probs
    df["Prediction"] = preds

    total = len(df)
    fraud = int(df["Prediction"].sum())
    safe = total - fraud

    fraud_percent = round((fraud / total) * 100, 2)

    # ---------------- METRICS ----------------
    st.markdown("### 📊 Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total", total)
    c2.metric("Fraud", fraud)
    c3.metric("Safe", safe)

    if fraud > 0:
        st.error(f"⚠ {fraud} Fraud Transactions ({fraud_percent}%)")
    else:
        st.success("✅ No Fraud Detected")

    # ---------------- CHARTS ----------------
    st.markdown("### 📈 Analytics")

    col1, col2 = st.columns(2)

    # DONUT CHART
    with col1:
        fig = go.Figure(data=[go.Pie(
            labels=["Safe", "Fraud"],
            values=[safe, fraud],
            hole=0.7,
            marker=dict(colors=["#00E396", "#FF4560"])
        )])

        fig.update_layout(
            height=350,
            annotations=[dict(
                text=f"{fraud_percent}%",
                x=0.5, y=0.5,
                font_size=26,
                showarrow=False
            )]
        )

        st.plotly_chart(fig, use_container_width=True)

    # HISTOGRAM
    with col2:
        fig2 = px.histogram(
            df,
            x="Fraud_Probability",
            nbins=50,
            color_discrete_sequence=["#FF4560"]
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------- FEATURE IMPORTANCE (FIXED) ----------------
    st.markdown("### 🔥 Top Features")

    try:
        importance = model.feature_importances_

        feat_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False).head(10)

        # 🔥 LOG SCALING FIX
        feat_df["Importance"] = np.log1p(feat_df["Importance"])

        fig3 = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues"
        )

        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)

    except:
        st.warning("Feature importance not available")

    # ---------------- FILTER ----------------
    st.markdown("### 📋 Data")

    if filter_option == "Fraud Only":
        df = df[df["Prediction"] == 1]
    elif filter_option == "Safe Only":
        df = df[df["Prediction"] == 0]

    st.dataframe(df.head(100), use_container_width=True)

else:
    st.info("Upload a dataset to start")
