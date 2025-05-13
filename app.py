# Créer le app.py complet avec le 4ᵉ onglet pour "Par Service (UF)"
app_with_uf_tab = """
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🧬 Tableau de bord unifié - Résistances bactériennes")

tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Antibiotiques 2024", 
    "🧪 Autres Antibiotiques", 
    "🧬 Phénotypes",
    "🏥 Par Service (UF)"
])

# Onglet 1
with tab1:
    st.header("📌 Antibiotiques - Données 2024")
    df_ab = pd.read_csv("tests_par_semaine_antibiotiques_2024.csv")
    df_ab.columns = df_ab.columns.str.strip()
    week_col = "Week" if "Week" in df_ab.columns else df_ab.columns[0]
    df_ab = df_ab[df_ab[week_col].apply(lambda x: str(x).isdigit())]
    df_ab[week_col] = df_ab[week_col].astype(int)
    ab_cols = [col for col in df_ab.columns if col.startswith('%')]
    selected_ab = st.selectbox("Antibiotique", ab_cols, key="ab2024")
    wmin, wmax = df_ab[week_col].min(), df_ab[week_col].max()
    w_range = st.slider("Plage de semaines", wmin, wmax, (wmin, wmax), key="range_ab2024")
    df_filtered = df_ab[df_ab[week_col].between(*w_range)]
    if week_col not in df_filtered.columns or selected_ab not in df_filtered.columns:
        st.error("❌ Erreur : Colonnes introuvables.")
    else:
        values = pd.to_numeric(df_filtered[selected_ab], errors='coerce').dropna()
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        low, high = max(q1 - 1.5 * iqr, 0), q3 + 1.5 * iqr
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_filtered[week_col], y=df_filtered[selected_ab],
                                 mode='lines+markers', name=selected_ab))
        fig.add_trace(go.Scatter(x=df_filtered[week_col], y=[high]*len(df_filtered),
                                 mode='lines', name="Seuil haut", line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=df_filtered[week_col], y=[low]*len(df_filtered),
                                 mode='lines', name="Seuil bas", line=dict(dash='dot')))
        fig.update_layout(yaxis=dict(range=[0, 30]), xaxis_title="Semaine", yaxis_title="Résistance (%)")
        st.plotly_chart(fig, use_container_width=True)

# Onglet 2
with tab2:
    st.header("🧪 Autres Antibiotiques - Staph aureus")
    df_other = pd.read_excel("other Antibiotiques staph aureus.xlsx")
    df_other.columns = df_other.columns.str.strip()
    week_col = "Week" if "Week" in df_other.columns else df_other.columns[0]
    df_other = df_other[df_other[week_col].apply(lambda x: str(x).isdigit())]
    df_other[week_col] = df_other[week_col].astype(int)
    ab_cols = [col for col in df_other.columns if col.startswith('%')]
    selected_ab = st.selectbox("Antibiotique", ab_cols, key="ab_other")
    wmin, wmax = df_other[week_col].min(), df_other[week_col].max()
    w_range = st.slider("Plage de semaines", wmin, wmax, (wmin, wmax), key="range_ab_other")
    df_filtered = df_other[df_other[week_col].between(*w_range)]
    if week_col not in df_filtered.columns or selected_ab not in df_filtered.columns:
        st.error("❌ Erreur : Colonnes introuvables.")
    else:
        values = pd.to_numeric(df_filtered[selected_ab], errors='coerce').dropna()
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        low, high = max(q1 - 1.5 * iqr, 0), q3 + 1.5 * iqr
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_filtered[week_col], y=df_filtered[selected_ab],
                                 mode='lines+markers', name=selected_ab))
        fig.add_trace(go.Scatter(x=df_filtered[week_col], y=[high]*len(df_filtered),
                                 mode='lines', name="Seuil haut", line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=df_filtered[week_col], y=[low]*len(df_filtered),
                                 mode='lines', name="Seuil bas", line=dict(dash='dot')))
        fig.update_layout(yaxis=dict(range=[0, 30]), xaxis_title="Semaine", yaxis_title="Résistance (%)")
        st.plotly_chart(fig, use_container_width=True)

# Onglet 3
with tab3:
    st.header("🧬 Phénotypes - Staphylococcus aureus")
    df = pd.read_excel("staph_aureus_pheno_final.xlsx")
    df.columns = df.columns.str.strip()
    df["week"] = pd.to_datetime(df["week"], errors="coerce")
    df = df.dropna(subset=["week"])
    df["Week"] = df["week"].dt.date
    phenos = ["MRSA", "Other", "VRSA", "Wild"]
    df["Total"] = df[phenos].sum(axis=1)
    for ph in phenos:
        df[f"% {ph}"] = (df[ph] / df["Total"]) * 100
    selected_pheno = st.selectbox("Phénotype", phenos, key="pheno")
    dmin, dmax = df["Week"].min(), df["Week"].max()
    d_range = st.slider("Plage de semaines", dmin, dmax, (dmin, dmax), key="range_pheno")
    filtered = df[df["Week"].between(*d_range)]
    col = f"% {selected_pheno}"
    values = filtered[col].dropna()
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    low, high = max(q1 - 1.5 * iqr, 0), q3 + 1.5 * iqr
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered["Week"], y=filtered[col],
                             mode='lines+markers', name=col))
    fig.add_trace(go.Scatter(x=filtered["Week"], y=[high]*len(filtered),
                             mode='lines', name="Seuil haut", line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=filtered["Week"], y=[low]*len(filtered),
                             mode='lines', name="Seuil bas", line=dict(dash='dot')))
    fig.update_layout(yaxis=dict(range=[0, 100]), xaxis_title="Semaine", yaxis_title="Résistance (%)")
    st.plotly_chart(fig, use_container_width=True)

# Onglet 4 : Par Service
with tab4:
    st.header("🏥 Résistance Staph aureus par service (UF)")
    df = pd.read_excel("staph aureus hebdomadaire excel.xlsx")
    df.columns = df.columns.str.strip()
    df["DATE_ENTREE"] = pd.to_datetime(df["DATE_ENTREE"], errors="coerce")
    df["Semaine"] = df["DATE_ENTREE"].dt.isocalendar().week
    ufs = df["DEMANDEUR"].dropna().astype(str).unique().tolist()
    selected_uf = st.selectbox("📍 Choisir un service / UF", ["Tous"] + sorted(ufs))
    if selected_uf != "Tous":
        df = df[df["DEMANDEUR"].astype(str) == selected_uf]
    ab_cols = ["Vancomycine", "Teicoplanine", "Gentamycine", "Oxacilline", "Daptomycine",
               "Dalbavancine", "Clindamycine", "Cotrimoxazole", "Linezolide"]
    selected_ab = st.selectbox("💊 Antibiotique", ab_cols)
    df = df.dropna(subset=["Semaine", selected_ab])
    df = df[df[selected_ab].isin(["S", "R"])]
    weekly = df.groupby("Semaine")[selected_ab].value_counts().unstack().fillna(0)
    weekly["Total"] = weekly.sum(axis=1)
    weekly["%R"] = 100 * weekly.get("R", 0) / weekly["Total"]
    values = weekly["%R"].dropna()
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = max(0, q1 - 1.5 * iqr), q3 + 1.5 * iqr
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly.index, y=weekly["%R"], mode='lines+markers', name=f"%R {selected_ab}"))
    fig.add_trace(go.Scatter(x=weekly.index, y=[high]*len(weekly), mode='lines', name="Seuil haut Tukey", line=dict(dash='dash', color='red')))
    fig.add_trace(go.Scatter(x=weekly.index, y=[low]*len(weekly), mode='lines', name="Seuil bas Tukey", line=dict(dash='dot', color='red')))
    fig.update_layout(xaxis_title="Semaine", yaxis_title="% Résistance", yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig, use_container_width=True)
"""

# Enregistrer le nouveau fichier
final_app_path = "/mnt/data/app.py"
with open(final_app_path, "w") as f:
    f.write(app_with_uf_tab)

final_app_path
