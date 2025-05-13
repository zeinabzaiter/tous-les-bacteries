# Générer un fichier app.py complet avec vérifications intégrées pour week_col et selected_ab
app_debugged_code = """
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🧬 Tableau de bord unifié - Résistances bactériennes")

tab1, tab2, tab3 = st.tabs([
    "📌 Antibiotiques 2024", 
    "🧪 Autres Antibiotiques", 
    "🧬 Phénotypes Staph aureus"
])

# === Onglet 1 ===
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
    st.write("📋 Colonnes disponibles :", df_filtered.columns.tolist())
    st.write("➡️ Semaine utilisée :", week_col)
    st.write("➡️ Antibiotique sélectionné :", selected_ab)

    if week_col not in df_filtered.columns or selected_ab not in df_filtered.columns:
        st.error("❌ Erreur : Colonnes introuvables dans les données.")
    else:
        values = pd.to_numeric(df_filtered[selected_ab], errors='coerce').dropna()
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        low, high = max(q1 - 1.5*iqr, 0), q3 + 1.5*iqr
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_filtered[week_col], y=df_filtered[selected_ab], mode='lines+markers', name=selected_ab))
        fig.add_trace(go.Scatter(x=df_filtered[week_col], y=[high]*len(df_filtered), mode='lines', name="Seuil haut", line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=df_filtered[week_col], y=[low]*len(df_filtered), mode='lines', name="Seuil bas", line=dict(dash='dot')))
        fig.update_layout(yaxis=dict(range=[0, 30]), xaxis_title="Semaine", yaxis_title="Résistance (%)")
        st.plotly_chart(fig, use_container_width=True)

# === Onglet 2 ===
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
    st.write("📋 Colonnes disponibles :", df_filtered.columns.tolist())
    st.write("➡️ Semaine utilisée :", week_col)
    st.write("➡️ Antibiotique sélectionné :", selected_ab)

    if week_col not in df_filtered.columns or selected_ab not in df_filtered.columns:
        st.error("❌ Erreur : Colonnes introuvables dans les données.")
    else:
        values = pd.to_numeric(df_filtered[selected_ab], errors='coerce').dropna()
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        low, high = max(q1 - 1.5*iqr, 0), q3 + 1.5*iqr
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_filtered[week_col], y=df_filtered[selected_ab], mode='lines+markers', name=selected_ab))
        fig.add_trace(go.Scatter(x=df_filtered[week_col], y=[high]*len(df_filtered), mode='lines', name="Seuil haut", line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=df_filtered[week_col], y=[low]*len(df_filtered), mode='lines', name="Seuil bas", line=dict(dash='dot')))
        fig.update_layout(yaxis=dict(range=[0, 30]), xaxis_title="Semaine", yaxis_title="Résistance (%)")
        st.plotly_chart(fig, use_container_width=True)

# Onglet 3 (inchangé)
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
"""

# Sauvegarder le fichier
app_debug_path = "/mnt/data/app_debugged.py"
with open(app_debug_path, "w") as f:
    f.write(app_debugged_code)

app_debug_path
