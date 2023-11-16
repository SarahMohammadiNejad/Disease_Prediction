import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np


from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures



# path = "/Users/sarah/Downloads/1-Sci/DataScience_Spiced/FinalProjects_Sara_MHMD/sarah2/data/processed"
# path = "../1-JNBs_Clean_DataSets/Clinical/data"

df_dropna = pd.read_csv("data/HDP_clinical_cleaned.csv")
'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
# A function to group variabls as ordinal, categorical, nuerical
'''
def variable_grouping(setting):
    setting.dropout_variables = []
    setting.categorical_variables = ['chest_pain_type','rest_ecg','st_slope','thalassemia','sex', 'fasting_blood_suger', 'exercise_induced_angina']
    setting.ordinal_variables = ['num_major_vessels','age','max_heart_rate_achieved','cholesterol','st_depression','resting_blood_pressure']
    setting.dependent_variables = ['heart_disease']

df_dropna['age'] = pd.cut(df_dropna['age'], bins=5)

'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
'''
def variable_desc():
    quest = {}
    quest = {'sex':"Are you male or female?",
             'age':'age',
            'cholesterol': "Enter your cholestrol", 
            'resting_blood_pressure':"Enter your blood pressure",
            'fasting_blood_suger':"Enter your blood suger",
            'chest_pain_type':'select the type of your chest pain',
            'rest_ecg': 'rest_ecg',
            'exercise_induced_angina':'exercise_induced_angina',
            'thalassemia':'thalassemia',
            'st_slope':'st_slope',
            'st_depression':'enter st_depression*10',
            'num_major_vessels':'num_major_vessels',
            'max_heart_rate_achieved':'max_heart_rate_achieved'
    }

    desc = {}
    desc = {'sex':['female','male'],
            'age':[0,120,1,30],
            'cholesterol': [100,700,1,200], 
            'resting_blood_pressure':[80,250,1,125],
            'fasting_blood_suger':['higher_than_120mg/ml', 'lower_than_120mg/ml'],
            'chest_pain_type':['typical_angina', 'asymptomatic', 'non_anginal_pain','atypical_angina'],
            'rest_ecg': ['left_ventricular_hypertrophy', 'normal', 'ST_T_wave_abnormality'],
            'exercise_induced_angina':['no', 'yes'],
            'thalassemia':['fixed_defect', 'normal', 'reversable_defect'],
            'st_slope':['downsloping', 'flat', 'upsloping'],
            'st_depression':[0,70,1,8],
            'num_major_vessels':['0', '3', '2', '1'],
            'max_heart_rate_achieved':[71,250,1,153]           
    }

    known_binary = {'sex':     {2: 'female', 1: 'male'},
                'exer': {1:'yes',2: 'no'},
                'diff_walk': {1:'yes',2: 'no'},
                'attack': {2: 0},
                'stroke': {2: 0},
                'angina': {2: 0},
                "skin_cancer": {2: 0},
                "other_cancer": {2: 0}
                }

    return quest, desc,known_binary

'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
# If in app.py "Sim.Create_EDAplot" is defined as True, 
this function creates the EDA plots
'''
def Create_EDA_Plots():
    font = {'family' : 'sans',
            'weight' : 'normal',
            'size'   : 12}

    matplotlib.rc('font', **font)

    plt.rc('font', size=10)          # controls default text sizes
    plt.rc('axes', titlesize=16)     # fontsize of the axes title
    plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
    feature_to_vis = ['sex','age','num_major_vessels','chest_pain_type','fasting_blood_suger','rest_ecg','exercise_induced_angina','st_slope','thalassemia']
    ncol = 3
    nrow = 3
    fig, ax = plt.subplots(nrow,ncol, figsize=(ncol*5,nrow*5))
    plt.subplots_adjust(wspace=0.25,hspace=0.25)
    for i_idx,i in enumerate(feature_to_vis):
        i_row = int(i_idx/ncol)
        i_col = i_idx%ncol
        sns.barplot(data = df_dropna, x = i, y = 'heart_disease', ax = ax[i_row][i_col])
        for tick in ax[i_row][i_col].get_xticklabels():
            tick.set_rotation(90)
            tick.set_ha('right')

    plt.tight_layout()
    plt.savefig('saved/Figs/EDA_Clinical.png')
    st.pyplot(fig)
'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
A function that read the data from cleaned dataset (csv file) 
and define the feature importnace proper for the dataset base on 
the observation fro EDA
'''
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

def func_read_fe(setting,sca):
    df_dropna = pd.read_csv("data/HDP_clinical_cleaned.csv")

    df_temp = df_dropna.copy()
    for i in setting.dropout_variables:
        df_temp = df_temp.drop([i],axis=1)
    for i in setting.dependent_variables:
        df_temp = df_temp.drop([i],axis=1)


    if sca=='MinMax': scaler = MinMaxScaler()
    elif sca=='Standard': scaler = StandardScaler()

    numerical_transformer1 = Pipeline(
        steps=[
            ('scaler',scaler)#,
            # ('scaler',MinMaxScaler())#,
            # ('polynomial', PolynomialFeatures(degree=2, include_bias=False))
            ]
            )

    categorical_transformer = Pipeline(
        steps=[
            ('ohe', OneHotEncoder(drop='first'))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer1, setting.ordinal_variables),
            ("cat", categorical_transformer, setting.categorical_variables)
        ],
        remainder = 'drop'
    )
    temp = preprocessor.fit_transform(df_temp)
    # ---------------
    feature_names = []

    # Add numerical feature names
    numerical_features = preprocessor.transformers_[0][2]  # Get numerical feature names
    numerical_feature_names = preprocessor.named_transformers_['num']['scaler'].get_feature_names_out(numerical_features)
    # numerical_feature_names = preprocessor.named_transformers_['num']['polynomial'].get_feature_names_out(numerical_features)
    feature_names.extend(numerical_feature_names)

    # Add categorical feature names
    categorical_features = preprocessor.transformers_[1][2]  # Get categorical feature names
    categorical_feature_names = preprocessor.named_transformers_['cat']['ohe'].get_feature_names_out(categorical_features)
    feature_names.extend(categorical_feature_names)
    # ---------------

    # feature_names = preprocessor.named_transformers_['ohe'].get_feature_names_out
    temp_fe = pd.DataFrame(temp,columns = feature_names,index = df_dropna.index)
    df_dropna_fe = pd.concat([temp_fe,df_dropna[setting.dependent_variables]],axis = 1)
    variables_to_process = df_temp.columns
    # ----------------------------------
    return preprocessor, df_dropna_fe,variables_to_process,feature_names

'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
The function that take the information from the user for learning features and then apply 
the feature engineering (based on the feature importance used for learning)
'''
def func_input_user(preprocessor,preprocessor_nn, variables_to_process,feature_names):


    selectbox_variables = ['sex', 'fasting_blood_suger','chest_pain_type','rest_ecg','st_slope','thalassemia', 'exercise_induced_angina','num_major_vessels']
    slider_variables = ['age','cholesterol','resting_blood_pressure','max_heart_rate_achieved','st_depression']
    questions, desc, known_binary = variable_desc()

    st.sidebar.write("**Interactive Element**")

    user_input2 = {}

    for i in slider_variables:
        user_input2[i] = st.sidebar.slider(f"{questions[i]}:",desc[i][0],desc[i][1], value=desc[i][3],step=desc[i][2])
    for i in selectbox_variables:
        rang = list(desc[i])
        user_input2[i] = st.sidebar.selectbox(f"{questions[i]}:\n{desc[i]}", rang)

    user_input2['st_depression'] = user_input2['st_depression']/10

    df_input = pd.DataFrame(user_input2,index = ['user_input'])
    df_input = df_input[variables_to_process]

    input_fe = preprocessor.transform(df_input)
    df_input_fe = pd.DataFrame(input_fe,columns = feature_names)

    input_fe_nn = preprocessor_nn.transform(df_input)
    df_input_fe_nn = pd.DataFrame(input_fe_nn,columns = feature_names)
    return df_input_fe,df_input_fe_nn

'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
This function first use randomundersampler to correct the imbalance 
in data and then splitting the data to train/test.
'''
def TrainTest(setting,df_dropna_fe,out):

    feature_cols = []
    for col in df_dropna_fe.columns:
        if col in setting.dependent_variables:

            continue
        feature_cols.append(col)

    
    X = df_dropna_fe[feature_cols]
    y = df_dropna_fe[out]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)


    return X_train, X_test, y_train, y_test


