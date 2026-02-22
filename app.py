import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# --- Configuration de la page ---
st.set_page_config(
    page_title="Système Anti-Fraude - Saint Jean Ingénieur", 
    layout="wide"
)

# --- Fonction de chargement du modèle ---
@st.cache_resource
def load_model():
    # Assure-toi que le fichier .pkl est dans le même dossier que app.py
    with open('modele_fraude_bancaire.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# --- Fonction principale de l'application ---
def main():
    st.title("💳 Détection de Fraude à la Carte de Crédit")
    st.write("Bienvenue sur l'interface de classification des transactions bancaires.")
    
    # Chargement du modèle
    try:
        model = load_model()
    except FileNotFoundError:
        st.error("Erreur : Le fichier 'modele_fraude_bancaire.pkl' est introuvable. Placez-le dans le répertoire courant.")
        return

    # Interface de chargement de fichier
    st.subheader("Analyse de lot de transactions")
    uploaded_file = st.file_uploader("Chargez un fichier de transactions au format CSV", type="csv")

    if uploaded_file is not None:
        # Lecture des données
        data = pd.read_csv(uploaded_file)
        st.write("Aperçu des données chargées :")
        st.dataframe(data.head())
        
        if st.button("Lancer la prédiction"):
            with st.spinner('Analyse par Random Forest en cours...'):
                
                # Isolation des features (on retire la colonne 'Class' si elle est présente par erreur dans le fichier de test)
                if 'Class' in data.columns:
                    X = data.drop('Class', axis=1)
                else:
                    X = data.copy()
                
                # Normalisation des données
                # Note : En production réelle, il faudrait charger le StandardScaler sauvegardé lors de l'entraînement. 
                # Ici, nous l'ajustons sur le lot pour la démonstration.
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Génération des prédictions
                predictions = model.predict(X_scaled)
                data['Prédiction (0=Normal, 1=Fraude)'] = predictions
                
                # Extraction des fraudes
                fraudes = data[data['Prédiction (0=Normal, 1=Fraude)'] == 1]
                
                st.success("Analyse terminée !")
                
                # Affichage conditionnel des résultats
                if len(fraudes) > 0:
                    st.error(f"⚠️ Alerte : {len(fraudes)} transaction(s) potentiellement frauduleuse(s) détectée(s) !")
                    st.write("Détail des transactions suspectes :")
                    st.dataframe(fraudes)
                else:
                    st.success("✅ Aucune transaction frauduleuse détectée dans ce lot.")

if __name__ == "__main__":
    main()