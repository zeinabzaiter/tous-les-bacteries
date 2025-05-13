# Générer le code d'une app Streamlit simple affichant les bactéries, leurs phénotypes et antibiotiques
bacteria_dashboard_code = """
import pandas as pd
import streamlit as st

# Charger les données
df = pd.read_excel("TOUS les bacteries a etudier.xlsx")
df.columns = df.columns.str.strip()

# Titre
st.title("🦠 Dashboard - Bactéries à surveiller")

# Barre de recherche
search = st.text_input("🔍 Rechercher une bactérie (ou laisser vide pour tout afficher)")

# Filtrage
if search:
    filtered_df = df[df['Category'].str.contains(search, case=False, na=False)]
else:
    filtered_df = df.copy()

# Affichage
st.dataframe(
    filtered_df.rename(columns={
        "Category": "Bactérie",
        "Phenotype": "Phénotypes",
        "Key Antibiotics": "Antibiotiques Clés",
        "Other Antibiotics": "Autres Antibiotiques"
    }),
    use_container_width=True
)
"""

# Sauvegarder le script dans un fichier
bacteria_dashboard_path = "/mnt/data/app_bacteries_surveillance.py"
with open(bacteria_dashboard_path, "w") as f:
    f.write(bacteria_dashboard_code)

bacteria_dashboard_path
