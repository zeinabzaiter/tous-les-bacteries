# Recréer le contenu propre de app.py (équivalent à app_dashboard_complet.py corrigé)
app_py_content = """
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🧬 Tableau de bord unifié - Résistances bactériennes")

tab1, tab2, tab3 = st.tabs(["📌 Antibiotiques 2024", "🧪 Autres Antibiotiques", "🧬 Phénotypes Staph aureus"])

with tab1:
    st.header("📌 Antibiotiques - Données 2024")
    df_ab = pd.read_csv("tests_par_semaine_antibiotiques_2024.csv")
    df_ab.columns = df_ab.columns.str.strip()
    week_col = "Week" if "Week" in df_ab.columns else df_ab.columns[0]
    df_ab = df_ab[df_ab[week_col].apply(lambda x: str(x).isdigit())]
    df_ab[week_col] = df_ab[week_col].astype(int)
    ab_cols = [col for col in df_ab.columns if col.startswith('%')]
    selected_ab = st.selectbox("Sélectionner un antibiotique", ab_cols, key="ab2024")
    min_week, max_week = df_ab[week_col].min(), df_ab[week_col].max()
    week_range = st.slider("Plage de semaines", min_week, max_week, (min_week, max_week), key="range_ab2024")
    df_filtered = df_ab[(df_ab[week_col] >= week_range[0]) & (df_ab[week_col] <= week_range[1])]
    values = pd.to_numeric(df_filtered[selected_ab], errors='coerce').dropna()
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower, upper = max(q1 - 1.5 * iqr, 0), q3 + 1.5 * iqr
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered[week_col], y=df_filtered[selected_ab], mode='lines+markers', name=selected_ab))
    fig.add_trace(go.Scatter(x=df_filtered[week_col], y=[upper]*len(df_filtered), mode='lines', name="Seuil haut", line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df_filtered[week_col], y=[lower]*len(df_filtered), mode='lines', name="Seuil bas", line=dict(dash='dot')))
    fig.update_layout(yaxis=dict(range=[0, 30]), xaxis_title="Semaine", yaxis_title="Résistance (%)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("🧪 Autres Antibiotiques - Staph aureus")
    df_other = pd.read_excel("other Antibiotiques staph aureus.xlsx")
    df_other.columns = df_other.columns.str.strip()
    week_col = "Week" if "Week" in df_other.columns else df_other.columns[0]
    df_other = df_other[df_other[week_col].apply(lambda x: str(x).isdigit())]
    df_other[week_col] = df_other[week_col].astype(int)
    other_cols = [col for col in df_other.columns if col.startswith('%')]
    selected_ab = st.selectbox("Sélectionner un antibiotique", other_cols, key="other_ab")
    min_week, max_week = df_other[week_col].min(), df_other[week_col].max()
    week_range = st.slider("Plage de semaines", min_week, max_week, (min_week, max_week), key="range_other_ab")
    df_filtered = df_other[(df_other[week_col] >= week_range[0]) & (df_other[week_col] <= week_range[1])]
    values = pd.to_numeric(df_filtered[selected_ab], errors='coerce').dropna()
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower, upper = max(q1 - 1.5 * iqr, 0), q3 + 1.5 * iqr
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered[week_col], y=df_filtered[selected_ab], mode='lines+markers', name=selected_ab))
    fig.add_trace(go.Scatter(x=df_filtered[week_col], y=[upper]*len(df_filtered), mode='lines', name="Seuil haut", line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df_filtered[week_col], y=[lower]*len(df_filtered), mode='lines', name="Seuil bas", line=dict(dash='dot')))
    fig.update_layout(yaxis=dict(range=[0, 30]), xaxis_title="Semaine", yaxis_title="Résistance (%)")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("🧬 Phénotypes - Staphylococcus aureus")
    df_pheno = pd.read_excel("staph_aureus_pheno_final.xlsx")
    df_pheno.columns = df_pheno.columns.str.strip()
    df_pheno["week"] = pd.to_datetime(df_pheno["week"], errors="coerce")
    df_pheno = df_pheno.dropna(subset=["week"])
    df_pheno["Week"] = df_pheno["week"].dt.date
    phenos = ["MRSA", "Other", "VRSA", "Wild"]
    df_pheno["Total"] = df_pheno[phenos].sum(axis=1)
    for pheno in phenos:
        df_pheno[f"% {pheno}"] = (df_pheno[pheno] / df_pheno["Total"]) * 100
    selected_pheno = st.selectbox("Sélectionner un phénotype", phenos, key="pheno")
    min_date, max_date = df_pheno["Week"].min(), df_pheno["Week"].max()
    date_range = st.slider("Plage de semaines", min_date, max_date, (min_date, max_date), key="range_pheno")
    filtered_pheno = df_pheno[(df_pheno["Week"] >= date_range[0]) & (df_pheno["Week"] <= date_range[1])]
    pct_col = f"% {selected_pheno}"
    values = filtered_pheno[pct_col].dropna()
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower, upper = max(q1 - 1.5 * iqr, 0), q3 + 1.5 * iqr
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_pheno["Week"], y=filtered_pheno[pct_col],
                             mode='lines+markers', name=f"% {selected_pheno}"))
    fig.add_trace(go.Scatter(x=filtered_pheno["Week"], y=[upper]*len(filtered_pheno),
                             mode='lines', name="Seuil haut", line=dict(dash='dash', color='red')))
    fig.add_trace(go.Scatter(x=filtered_pheno["Week"], y=[lower]*len(filtered_pheno),
                             mode='lines', name="Seuil bas", line=dict(dash='dot', color='red')))
    fig.update_layout(yaxis=dict(range=[0, 100]), xaxis_title="Semaine", yaxis_title="Résistance (%)")
    st.plotly_chart(fig, use_container_width=True)
"""

# Enregistrer sous app.py
app_fixed_path = "/mnt/data/app.py"
with open(app_fixed_path, "w") as f:
    f.write(app_py_content)

app_fixed_path
