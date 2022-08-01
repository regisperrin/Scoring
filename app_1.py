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
    sample = pd.read_csv('archive/df_clients_to_predict_feature_selected_id.csv', index_col='SK_ID_CURR', encoding ='utf-8')
    
    description = pd.read_csv("archive/features_description.csv", 
                              usecols=['Row', 'Description'], index_col=0, encoding= 'unicode_escape')

    target = data.iloc[:, -1:]

    return data, sample, target, description

def load_sample():
              
    sample = pd.read_csv('archive/df_clients_to_predict_feature_selected_id.csv', index_col='SK_ID_CURR', encoding ='utf-8')

    return sample

def load_model():
    '''loading the trained model'''
    #pickle_in = open('archive/model_1.pkl', 'rb') 
    #clf = pickle.load(pickle_in)
    
    # Loading the model
    clf = joblib.load('archive/model_1.pkl')
    
    return clf


@st.cache(allow_output_mutation=True)
def load_knn(sample):
    knn = knn_training(sample)
    return knn


@st.cache
def load_infos_gen(data):
    lst_infos = [data.shape[0],
                 round(data["AMT_INCOME_TOTAL"].mean(), 2),
                 round(data["AMT_CREDIT"].mean(), 2)]

    nb_credits = lst_infos[0]
    rev_moy = lst_infos[1]
    credits_moy = lst_infos[2]

    targets = data.TARGET.value_counts()

    return nb_credits, rev_moy, credits_moy, targets


def identite_client(data, id):
    data_client = data[data.index == int(id)]
    return data_client

@st.cache
def load_age_population(data):
    data_age = round((data["DAYS_BIRTH"]/365), 2)
    return data_age

@st.cache
def load_income_population(sample):
    df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
    df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
    return df_income

@st.cache
def load_prediction(sample, id, clf):
    #X=sample.iloc[:, :-1]
    X = sample
    score = clf.predict_proba(X[X.index == int(id)])[:,1]
    return score

@st.cache
def load_kmeans(sample, id, mdl):
    index = sample[sample.index == int(id)].index.values
    index = index[0]
    data_client = pd.DataFrame(sample.loc[sample.index, :])
    df_neighbors = pd.DataFrame(knn.fit_predict(data_client), index=data_client.index)
    df_neighbors = pd.concat([df_neighbors, data], axis=1)
    return df_neighbors.iloc[:,1:].sample(10)

@st.cache
def knn_training(sample):
    knn = KMeans(n_clusters=2).fit(sample)
    return knn 





app_mode = st.sidebar.selectbox('Select Page',['Home','Predict'])

if app_mode=='Home':
    st.title('Customer Prediction')


elif app_mode == 'Predict':

    st.subheader('Fill in Customer ID to get prediction ')
    st.sidebar.header("Other details :")
    
    #Loading data……
    sample = load_sample()
    id_client = sample.index.values
    clf = load_model()

    #Loading selectbox
    chk_id = st.selectbox("Client ID", id_client)

    if st.button("Predict"):

        st.header("**Customer file analysis**")
        prediction = load_prediction(sample, chk_id, clf)
        st.write("**Default probability : **{:.0f} %".format(round(float(prediction)*100, 2)))


