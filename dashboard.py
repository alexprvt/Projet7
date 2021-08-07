"""
@author: Alexandre
"""

import streamlit as st
from PIL import Image
import pandas as pd

from graphs import default_gauge
from graphs import plotly_waterfall
from graphs import plot_bar_default
from graphs import plot_bars
  
URL = 'http://localhost:8501'
path = 'C:\\Users\\Alexandre\\Desktop\\Dashboard_Test\\'

def main():
    
    #--------------------------------------------------------------------------
    ### Fonctions d'obtention d'informations des prêts dans la base de données
    #--------------------------------------------------------------------------
    
    @st.cache
    def get_ids():
        #chargement de la table
        score_table = pd.read_csv(path + 'score_table.csv')
        ids = score_table.SK_ID_CURR.values[:1000]
        return ids
    
    
    @st.cache
    def get_score(sk_id):
        #chargement de la table
        score_table = pd.read_csv(path + 'score_table.csv')
        score = score_table.set_index('SK_ID_CURR').loc[sk_id].SCORE
        return score
    
    
    @st.cache
    def get_target(sk_id):
        #chargement de la table
        score_table = pd.read_csv(path + 'score_table.csv')
        target = score_table.set_index('SK_ID_CURR').loc[sk_id].TARGET
        if target == 0:
            decision = 'Crédit accordé'
        else:
            decision = 'Crédit refusé' 
        return decision, target
    
    @st.cache
    def get_shap_values(sk_id):
        #chargement de la table des shap_values
        shap_data = pd.read_csv(path+'shap_data_1000.csv')
        shap_series = shap_data.set_index('Feature')[str(sk_id)]
        return shap_series
    
    @st.cache
    def load_data():
        #chargement des données
        data = pd.read_csv(path+'dashboard_data.csv')
        data = data.set_index('SK_ID_CURR')
        return data
    
    #--------------------------------------------------------------------------
    ### Configuration de la page streamlit
    #--------------------------------------------------------------------------
    
    # Chargement du logo à placer dans l'onglet de la page
    logo_AP = Image.open(path+'\\logo_AP.png')       
    
    st.set_page_config(
         # permet à l'app web de prendre tout l'écran
        layout="wide", 
        # titre de la page                             
        page_title="Dashoard: Prêt à Dépenser",  
        # configuration du logo de la page
        page_icon=logo_AP,
        # impose l'ouverture de labarre d'options au lancement de l'app                       
        initial_sidebar_state="expanded",            
        )
    
    # Ajout du titre du Dashboard intéractif
    st.title("__*Dashboard intéractif: \
             compréhension de la capacité de remboursement d'un crédit*__")
     # Ajout du  lien vers le repository Github du projet
    st.markdown('Alexandre PRÉVOT - \
              *[Lien vers le dépôt Github](https://github.com/alexprvt)*')
    
    #Ajout d'une barre d'option latérale
    st.sidebar.title("Options d'affichage")
    
    
    #--------------------------------------------------------------------------
    ### Sélection de l'ID du prêt parmi les identifiants de la base de données
    #--------------------------------------------------------------------------
    
    IDS = get_ids()
    select_id = st.sidebar.selectbox('Identifiant du crédit: ', IDS)
    
    
    #--------------------------------------------------------------------------
    ### Sélection du type d'affichage
    #--------------------------------------------------------------------------
    
    menu=['Capacité de remboursement',
          'Informations relatives au client']
    
    radio = st.sidebar.radio("Renseignements", menu)
    
    
    #--------------------------------------------------------------------------
    ### Renseignements sur la capacité de remboursement du prêt
    #--------------------------------------------------------------------------
    
    if radio == 'Capacité de remboursement':
        st.header("__Prédiction de capacité à rembourser le prêt__")
        
        
        #Calcul de la probabilité de difficulté de paiement
        default_proba = int(100*(1-get_score(select_id)))
        fig = default_gauge(default_proba)
        
        #Séparation de l'espace en deux colonnes
        left_col, right_col = st.beta_columns(2)
        
        #Jauge de probabilité de difficulté de paiement
        left_col.plotly_chart(fig, use_container_width=True)
        
        right_col.write("_Lorsque la probabilité qu'il y ait un défaut de paiement \
                 dépasse 50%, l'algorithme prédit que le prêt ne doit pas être\
                 accordé. Cependant le modèle de machine learning utilisé \
                 n'est pas rigouresement fiable. Après entrâinement de ce \
                 modèle, 70% des prêts ont été correctement classifiés._")
        
        #Permet le choix du seuil de probabilité de défaut admissible
        seuil = right_col.slider(
                "Seuil d'acceptabilité",
                min_value=1,
                max_value=99,
                value=50,
                step=1,
                help="Probabilité maximale du risque de défaut de paiement \
                acceptable pour accorder le prêt"
                )
        
        if default_proba <= seuil:
            decision = "Crédit accordé"
            emoji = "white_check_mark"
            
        else:
            decision = "Crédit refusé"
            emoji = "no_entry_sign"
          
        right_col.header(f"***Décision:*** {decision} :{emoji}:")
        
        expander = right_col.beta_expander("En réalité, le client a-t-il eu du mal \
                                    à rembourser son prêt ?")
        
        # Bandeau de vérification de la prédiction
        reality = get_target(select_id)[0]
        if reality == "Crédit refusé":
            expander.write("Le client a eu des difficultés à rembourser ce \
                           prêt :no_entry_sign: ")
        else:
            expander.write("Le client n'a pas eu de difficultés à rembourser ce \
                           prêt :white_check_mark: ")
        if reality != decision:
            expander.write("La prédiction du modèle n'est donc pas correcte")
        else :
            expander.write("La prédiction du modèle est donc correcte")
        
        
        st.header('__Interprétabilité de la prédiction de défaut de paiement__')
        left_col, right_col = st.beta_columns(2)
        
        #Selection du nombre de top variables à afficher pour le waterfall plot
        n_feats = left_col.slider(
                "Nombre de variables les plus influentes sur la prédiction du modèle",
                min_value=5,
                max_value=30,
                value=10,
                step=1,
                help="Sélectionnez entre 5 et 30 variables à afficher"
                )
        
        #Affichage du waterfall plot pour l'interprétabilité de la prédiction
        shap_series = get_shap_values(select_id)
        fig = plotly_waterfall(select_id, shap_series, n_feats=n_feats)[0]
        left_col.plotly_chart(fig, use_container_width=True)
        
        #Choix de la variable à afficher dans le graphique
        VARS = plotly_waterfall(select_id, shap_series, n_feats=n_feats)[1] 
        var = right_col.selectbox('Sélection de la variable à afficher', VARS)
        
        #Cargement des données
        data = load_data()
        
        if data[var].nunique() > 2:
            
            #Selection du nombre de top variables à afficher pour le waterfall plot
            n_bins = right_col.slider(
                    "Nombre de barres du graphique ci-dessous",
                    min_value=2,
                    max_value=110,
                    value=10,
                    step=1,
                    help="Sélectionnez entre 2 et 100 barres à afficher"
                    )
            #Affichage du graphique
            bars = plot_bar_default(data, var, select_id, n_bins=n_bins)
            right_col.plotly_chart(bars, use_container_width=True)
            
        else:
            #Affichage du graphique
            bars = plot_bars(data, var, select_id)[0]
            right_col.plotly_chart(bars, use_container_width=True)

if __name__ == '__main__':
    main()