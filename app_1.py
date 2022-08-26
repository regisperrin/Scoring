import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap
import plotly.express as px
from zipfile import ZipFile
from sklearn.cluster import KMeans
plt.style.use('fivethirtyeight')
#sns.set_style('darkgrid')

import lightgbm as lgb
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import joblib

@st.cache
def load_data():
    #z = ZipFile("data/archive_selected.zip")
    #data = pd.read_csv(z.open('default_risk.csv'), index_col='SK_ID_CURR', encoding ='utf-8')
    #sample = pd.read_csv(z.open('X_sample.csv'), index_col='SK_ID_CURR', encoding ='utf-8')
    
    data = pd.read_csv('archive/df_current_clients.csv', index_col='SK_ID_CURR', encoding ='utf-8')
    sample = pd.read_csv('archive/df_clients_to_predict_sf.csv', index_col='SK_ID_CURR', encoding ='utf-8')
    
    description = pd.read_csv("archive/features_description.csv", 
                              usecols=['Row', 'Description'], index_col=0, encoding= 'unicode_escape')

    target = data.iloc[:, -1:]

    return data, sample, target, description

def load_sample():      
    sample = pd.read_csv('archive/df_clients_to_predict_sf.csv', index_col='SK_ID_CURR', encoding ='utf-8')
    return sample

def load_model():
    # Loading the model
    obj = joblib.load('archive/obj_model.pkl')   
    return obj

def load_scaler():
    # Loading the model
    obj = joblib.load('archive/obj_scaler.pkl')
    return obj

def load_shap_vals():
    # Loading the model
    obj = joblib.load('archive/obj_shap_vals.pkl')
    return obj


def get_data_client(df, id):
        data_client = df[df.index == int(id)]
        return data_client


@st.cache
def load_score(sample, id, model):
    X = sample
    score = model.predict_proba(X[X.index == int(id)])[:,1]
    return score

def load_prediction(sample, id, model):    
    X = sample    
    client_scaled = scaler.transform(X[X.index == int(id)].values.reshape(1, -1))    
    proba_client = model.predict_proba(client_scaled)
    return proba_client

def get_pos(sample, id):
    pos = sample.index.get_loc(int(id))
    return pos

app_mode = st.sidebar.selectbox('Select Page',['Home','Predict'])
    
if app_mode=='Home':
    st.title('Customer Prediction')


elif app_mode == 'Predict':

    st.subheader('Fill in Customer ID to get prediction ')
    st.sidebar.header("Other details :")
    
    #Loading data……
    sample = load_sample()
    id_client = sample.index.values
    model = load_model()
    scaler  = load_scaler()
    shap_vals = load_shap_vals()

    threshold = 0.152

    #Loading selectbox
    chk_id = st.selectbox("Client ID", id_client)
    
    pos= 0

    if st.button("Predict"):

        st.header("Analyse du Client")
        
        
        
        proba_client = load_prediction(sample, chk_id, model)
        score = load_score(sample, chk_id, model)
        
        y_prob = proba_client[:, 1]

        res = (y_prob >= threshold).astype(int)

        if (int(res[0]) == 0):
            res = "Oui"
        else:
            res = "Non"
        
        st.write("Resultat : ",res)
        st.write("Proba_0 : ",proba_client[0][0])
        st.write("Proba_1 : ",proba_client[0][1])
        
        st.write("Default probability : {:.0f} %".format(round(float(score)*100, 2)))
        
        
        st.write("Pos : ", pos)


    if st.checkbox("Informations du client :"):

        data_client = get_data_client(sample, chk_id)

        st.write("Age : {:.0f} ans".format(int(data_client["DAYS_BIRTH"]/365)))

    if st.checkbox("Customer ID {:.0f} explications ?".format(chk_id)):
        shap.initjs()
        X = sample
        X = X[X.index == chk_id]
        number = st.slider("Nombre de features", 0, 20, 5)
                
        fig, ax = plt.subplots(figsize=(10, 10))        
        shap.bar_plot(shap_vals[pos],
              feature_names=sample.columns,
              max_display=number)
        st.pyplot(fig)
        
        
        
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)


