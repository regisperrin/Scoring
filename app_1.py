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
 
# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

@st.cache
def load_data():
    #z = ZipFile("data/archive_selected.zip")
    
    data = pd.read_csv('archive/df_current_clients.csv', index_col='SK_ID_CURR', encoding ='utf-8')
    predict = pd.read_csv('archive/df_clients_to_predict_sf.csv', index_col='SK_ID_CURR', encoding ='utf-8')
    
    description = pd.read_csv("archive/features_description.csv", 
                              usecols=['Row', 'Description'], index_col=0, encoding= 'unicode_escape')

    target = data.iloc[:, -1:]

    return data, predict, target, description

@st.cache
def load_predict():      
    predict = pd.read_csv('archive/df_clients_to_predict_sf.csv', index_col='SK_ID_CURR', encoding ='utf-8')
    return predict

@st.cache
def load_model():
    # Loading the model
    obj = joblib.load('archive/obj_model.pkl')   
    return obj

@st.cache
def load_scaler():
    # Loading the model
    obj = joblib.load('archive/obj_scaler.pkl')
    return obj

@st.cache
def load_shap_vals():
    # Loading the model
    obj = joblib.load('archive/obj_shap_vals.pkl')
    return obj

def get_data_client(df, id):
        data_client = df[df.index == int(id)]
        return data_client

def load_score(df, id, model):
    score = model.predict_proba(df[df.index == int(id)])[:,1]
    return score

def load_prediction(df, id, model):    
    X = df    
    client_scaled = scaler.transform(X[X.index == int(id)].values.reshape(1, -1))    
    proba_client = model.predict_proba(client_scaled)
    return proba_client

def get_pos(df, id):
    pos = df.index.get_loc(int(id))
    return pos


# sidebar

title = """
<div style="background-color: white; padding:10px; border-radius:10px">
<h1 style="color: black; text-align:center">Dashboard Prêt à dépenser</h1>
</div>
"""
st.markdown(title, unsafe_allow_html=True)

#data
predict = load_predict()
client_list = predict.index.values
model = load_model()
scaler  = load_scaler()
shap_vals = load_shap_vals()
threshold = 0.152

# Check if 'key' already exists in session_state
# If not, then initialize it
if 'pred' not in st.session_state:
    st.session_state['pred'] = 0

# 3 columns layout
c1, c2,c3 = st.columns((1,2,1))

#Loading selectbox
chk_id = c1.selectbox("Choix Client", client_list)

if c1.button("Prediction Score") or st.session_state['pred'] == chk_id:

    st.session_state['pred'] = chk_id
    pos = get_pos(predict, chk_id)
    c1.subheader("Score pour client: " + str(chk_id) )
    
        
    proba_client = load_prediction(predict, chk_id, model)
    score = load_score(predict, chk_id, model)
    
    y_prob = proba_client[:, 1]

    res = (y_prob >= threshold).astype(int)

    if (int(res[0]) == 0):
        res = "Oui"
    else:
        res = "Non"


    c1.write("Resultat: " + str(res))
    c1.write("Proba_0 : "+ str(proba_client[0][0]))
    c1.write("Proba_1 : " + str(proba_client[0][1]))
    
    c1.write("Default probability : {:.0f} %".format(round(float(score)*100, 2)))

    data_client = get_data_client(predict, chk_id)

    c1.subheader("Informations du client")

    c1.write("Age : {:.0f} ans".format(abs(int(data_client["DAYS_BIRTH"]/365))))
    c1.write("Work : {:.0f} ans".format(abs(int(data_client["DAYS_EMPLOYED"]/365))))

if c2.checkbox("Customer ID {:.0f} explications ?".format(chk_id)):
    shap.initjs()
    X = predict
    X = X[X.index == chk_id]
    number = c2.slider("Nombre de features",5, 20, 10)
    
    fig, ax = plt.subplots(figsize=(10, number/2))        
    shap.bar_plot(shap_vals[pos],
          feature_names=predict.columns,
          max_display=number)
    c2.pyplot(fig)
    
    
    
else:
    st.markdown("<i></i>", unsafe_allow_html=True)


