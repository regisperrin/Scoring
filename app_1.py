import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap
import plotly.express as px
from zipfile import ZipFile

plt.style.use('fivethirtyeight')
#sns.set_style('darkgrid')

import lightgbm as lgb
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import joblib
from datetime import date, timedelta
import plotly.figure_factory as ff
import plotly.graph_objs as go
import ipywidgets as widgets

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

#shap.initjs()

def calculate_years(days):
    today = date.today()
    initial_date = today - timedelta(abs(days))
    years = today.year - initial_date.year - ((today.month, today.day) < (initial_date.month, initial_date.day))

    return years

@st.cache
def load_data():
    #z = ZipFile("data/archive_selected.zip")
    
    df_current_clients = pd.read_csv('archive/df_current_clients.csv', index_col='SK_ID_CURR', encoding ='utf-8')

    df_current_clients["AGE"] = df_current_clients["DAYS_BIRTH"].apply(lambda x: calculate_years(x))
    df_current_clients["YEARS_EMPLOYED"] = df_current_clients["DAYS_EMPLOYED"].apply(lambda x: calculate_years(x))

    df_current_clients_repaid = df_current_clients[df_current_clients["TARGET"] == 0]
    df_current_clients_not_repaid = df_current_clients[df_current_clients["TARGET"] == 1]

    
    #description = pd.read_csv("archive/features_description.csv", 
    #                          usecols=['Row', 'Description'], index_col=0, encoding= 'unicode_escape')

    return df_current_clients_repaid, df_current_clients_not_repaid

@st.cache
def load_predict():      
    predict = pd.read_csv('archive/df_clients_to_predict_sf.csv', index_col='SK_ID_CURR', encoding ='utf-8')
    return predict

@st.cache
def load_model():
    # Loading the model
    obj = joblib.load('archive/obj_model.pkl')   
    return obj

def load_stats():
    # Loading the model
    obj = joblib.load('archive/obj_figs.pkl')
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
        data_client["AGE"] = data_client["DAYS_BIRTH"].apply(lambda x: calculate_years(x))
        data_client["YEARS_EMPLOYED"] = data_client["DAYS_EMPLOYED"].apply(lambda x: calculate_years(x))

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

def get_shap_feature_client(df, shap_vals, nb=20) :
    feature_names = df.columns
    shap_importance = pd.DataFrame(list(zip(feature_names, shap_vals)),
                                      columns=['feature','importance_normalized'])
    shap_importance["importance_normalized"] = abs(shap_importance["importance_normalized"])
    shap_importance.sort_values(by=['importance_normalized'],ascending=False, inplace=True)
    return shap_importance.head(nb)

# sidebar

title = """
<div style="background-color: white; padding:10px; border-radius:10px">
<h1 style="color: black; text-align:center">Dashboard Prêt à dépenser</h1>
</div>
"""
st.markdown(title, unsafe_allow_html=True)

#data
#df_current_clients_repaid, df_current_clients_not_repaid = load_data()
predict = load_predict()
client_list = predict.index.values
model = load_model()
scaler  = load_scaler()
shap_vals = load_shap_vals()
threshold = 0.6

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
    
    oui_html = """
    <div style="background-color: green;margin:  0px 20px 20px 20px; border-radius:10px">
    <h3 style="color: white; text-align:center">OUI</h3>
    </div>
    """
    non_html = """
    <div style="background-color: red; margin: 0px 20px 20px 20px; border-radius:10px">
    <h3 style="color: white; text-align:center">NON</h3>
    </div>
    """

    if (int(res[0]) == 0):
        c1.markdown(oui_html, unsafe_allow_html=True)
    else:
        c1.markdown(non_html, unsafe_allow_html=True)


    #c1.write("Resultat: ")
    
    c1.write("Proba_0 : "+ str(proba_client[0][0]))
    c1.write("Proba_1 : " + str(proba_client[0][1]))
    c1.write("Seuil : " + str(threshold))
    #c1.write("Default probability : {:.0f} %".format(round(float(score)*100, 2)))

    data_client = get_data_client(predict, chk_id)

    c1.subheader("Informations du client")

    Genre = "Femme"
    if int(data_client["CODE_GENDER"]) == 1 :
        Genre = "Homme"
        
    c1.write("Age : {:.0f} ans".format(int(data_client["AGE"])))
    c1.write("Genre : " + Genre)
    c1.write("Travail : {:.0f} ans".format(int(data_client["YEARS_EMPLOYED"])))
    c1.write("Enfants : {:.0f} ".format(int(data_client["CNT_CHILDREN"])))
    c1.write("Revenu : {:.0f} ".format(int(data_client["AMT_INCOME_TOTAL"])))
    c1.write("Crédit : {:.0f} ".format(int(data_client["AMT_CREDIT"])))
    c1.write("Rente : {:.0f} ".format(int(data_client["AMT_ANNUITY"])))
    c1.write("Biens : {:.0f} ".format(int(data_client["AMT_GOODS_PRICE"])))

#if c2.checkbox("Customer ID {:.0f} explications ?".format(chk_id)):
    
    c2.subheader("Customer ID {:.2f} explications ".format(chk_id))
    
    pos = get_pos(predict, chk_id)
    
    number = c2.slider("Nombre de features",5, 20, 10)
    
    fig, ax = plt.subplots(figsize=(10, number/2))        
    shap.bar_plot(shap_vals[pos],
          feature_names=predict.columns,
          max_display=number)
    c2.pyplot(fig)
    
    shap_importance = get_shap_feature_client(predict, shap_vals[pos],number)
    col_list =list(shap_importance["feature"])
    c2.table(data_client[col_list].transpose())
    
    stats_list = load_stats()
    
    group_labels = ["Repaid", "Not repaid"]
    colors=["Green", "Red"]
    
    fig_feature_list = ['EXT_SOURCE_3','EXT_SOURCE_2', 'PAYMENT_RATE', 'AGE', 'YEARS_EMPLOYED',
                    'AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY']
    
    for feature in fig_feature_list:
        fig = stats_list[feature] 
        fig.add_vline(x=float(data_client[feature]), line_width=3,
                         line_dash="dash", line_color="blue", annotation_text="Client")
        c3.plotly_chart(fig)
    
    # Create distplot
#     fig_AMT_CREDIT = stats_list["AMT_CREDIT"] 
#     fig_AMT_CREDIT.update_layout(
#                         paper_bgcolor="white",
#                         font={
#                             "family": "Source Sans Pro"
#                         },
#                         autosize=False,
#                         width=500,
#                         height=360,
#                         margin=dict(
#                             l=50, r=50, b=0, t=20, pad=0
#                         ),
#                         title={
#                             "text" : "EXT_SOURCE_3",
#                             "y" : 1,
#                             "x" : 0.45,
#                             "xanchor" : "center",
#                             "yanchor" : "top"
#                         },
#                         xaxis_title="amt_credit",
#                         yaxis_title="density",
#  
#                     )
#     fig_AMT_CREDIT.add_vline(x=float(data_client["EXT_SOURCE_3"]), line_width=3,
#                          line_dash="dash", line_color="blue", annotation_text="Client")
#     c3.plotly_chart(fig_AMT_CREDIT)


else:
    st.markdown("<i></i>", unsafe_allow_html=True)


