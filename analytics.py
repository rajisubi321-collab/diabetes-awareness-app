from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

def render_analytics(results_path: Path, data_path: Path):
    st.header("📊 Predictive analytics")
    st.caption("Metrics describe performance on a held-out test split from the source dataset; they do not guarantee performance in other populations.")
    if results_path.exists():
        results = pd.read_csv(results_path)
        if not results.empty:
            row = results.iloc[0]
            cols = st.columns(5)
            for col, key, label, fmt in zip(
                cols,
                ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"],
                ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
                ["percent", "percent", "percent", "percent", "decimal"]
            ):
                value = float(row[key])
                col.metric(label, f"{value * 100:.1f}%" if fmt == "percent" else f"{value:.3f}")
            st.dataframe(results, use_container_width=True, hide_index=True)
    else:
        st.info("Performance report not found. Run `python data/train_main.py` first.")

    if data_path.exists():
        try:
            df = pd.read_csv(data_path)
            st.subheader("Source dataset patterns")
            if "Diabetes_binary" in df.columns:
                counts = df["Diabetes_binary"].value_counts().rename_axis("Class").reset_index(name="Records")
                counts["Class"] = counts["Class"].map({0: "Class 0", 1: "Class 1"}).fillna(counts["Class"].astype(str))
                st.plotly_chart(px.bar(counts, x="Class", y="Records", title="Target class distribution"), use_container_width=True)
            if "BMI" in df.columns:
                st.plotly_chart(px.histogram(df, x="BMI", nbins=35, title="BMI distribution in source dataset"), use_container_width=True)
            if "PhysActivity" in df.columns:
                activity = df["PhysActivity"].map({0: "No activity", 1: "Activity"}).value_counts().rename_axis("Physical activity").reset_index(name="Records")
                st.plotly_chart(px.bar(activity, x="Physical activity", y="Records", title="Physical activity in source dataset"), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not load dataset charts: {exc}")
