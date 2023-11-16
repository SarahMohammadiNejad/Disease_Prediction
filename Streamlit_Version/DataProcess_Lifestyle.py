import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df_dropna = pd.read_csv("data/Lifestyle2021_Clean.csv")
'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
# A function to group variabls as ordinal, categorical, nuerical
'''
def variable_grouping(setting):
    setting.dropout_variables = ['BMI_cat','CHILDREN','phys_ment_poor30']
    setting.categorical_variables= ['sex',"cholestrol","blood_pressure",'exer','diff_walk','marital_status','employ_status','diabetes']
    setting.dependent_variables = ['attack','stroke','angina',"skin_cancer","other_cancer"]
    setting.ordinal_variables = list(set(df_dropna.columns) - set(setting.dropout_variables)- set(setting.categorical_variables)-set(setting.dependent_variables))
'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
'''
def variable_desc():
    quest = {}
    quest = {'sex':"Are you male or female?",
            'cholestrol': "Ever Told Cholesterol Is High", 
            'blood_pressure':"Ever Told Blood Pressure High",
            "diabetes":"Ever told you had diabetes",
            'exer':"Exercise in Past 30 Days",
            'diff_walk':"Difficulty Walking or Climbing Stairs",
            'marital_status': "Marital Status",
            'employ_status': "Employment Status",
            'BMI5':'Computed body mass index*100',
            'income':"Income Level",
            'mental_poor30':'#days during the past 30 your mental health not good?',
            'phys_ment_poor30':'#days during the past 30 your physical or mental health not good?',
            'smoke':'Frequency of Days Now Smoking',
            'CHILDREN':'Number of Children in Household',
            'checkup':'time since last routine checkup',
            'alcohol30':'#days/month alcoholic beverage',
            'EDUCA':"Education Level",
            'age_cat5':"age in five-year age categories"
    }
    AXlabel = {}
    AXlabel = {
            'cholestrol':'cholesterol',
            'blood_pressure': 'blood pressure',
            'age_cat5':'age',
            'EDUCA':'education',
            'BMI_cat':'BMI',
            'employ_status':'employ status',
            'marital_status':'marital status',
            "diabetes":"diabetes"
    }

    quest = {'sex':"Are you male or female?",
            'cholestrol': "Ever Told Cholesterol Is High", 
            'blood_pressure':"Ever Told Blood Pressure High",
            "diabetes":"Ever told you had diabetes",
            'exer':"Exercise in Past 30 Days",
            'diff_walk':"Difficulty Walking or Climbing Stairs",
            'marital_status': "Marital Status",
            'employ_status': "Employment Status",
            'BMI5':'Computed body mass index*100',
            'income':"Income Level",
            'mental_poor30':'#days during the past 30 your mental health not good?',
            'phys_ment_poor30':'#days during the past 30 your physical or mental health not good?',
            'smoke':'Frequency of Days Now Smoking',
            'CHILDREN':'Number of Children in Household',
            'checkup':'time since last routine checkup',
            'alcohol30':'#days/month alcoholic beverage',
            'EDUCA':"Education Level",
            'age_cat5':"age in five-year age categories"
    }
    desc = {}
    desc = {'sex':{2: 'female', 1: 'male'},
            'cholestrol': {1:'yes',2: 'no'}, 
            'blood_pressure':{1:'yes',2: 'y preg',3:'no',4:'borderl'},
            "diabetes":{1:'yes',2: 'y preg',3:'no',4:'borderl'},
            'exer':{1:'yes',2: 'no'},
            'diff_walk':{1: 'yes', 2:'no'},
            'marital_status': {1:'married',2:'divorced',3:'widowed',4:'seperated',5:'nev mar',6:'unmar cupl'},
            'employ_status': {1:'employed',2:'self.empl',3:'out>1y',4:'out<1y',5:'homemkr',6:'student',7:'retired',8:'unable'},
            'BMI5':{'1-9999'},
            'BMI_cat':{1:'Uweight',2:'normal',3:'Oweight',4:'obese'},
            'income':{1:'<10K',2:'10K<<15K',3:'15<<20K',4:'20<<25K',5:'25<<35K',6:'35<<50K',7:'50<<75K',8:'75<<100K',9:'100<<150K',10:'150<<200K',11:'>200k'},
            'mental_poor30':{'1<<30'},
            'phys_ment_poor30':{'1<<30'},
            'smoke':{1:"everyDay",2:'someDays',3:'never'},
            'CHILDREN':{'<88'},
            'checkup':{1: '<12m', 2:'1y<<2y',3: '2y<<5y', 4:'>5y'},
            'alcohol30':{'200<<230'},
            'EDUCA':{1: 'Never',2: "G1-8",3: "G9-11",4: "G12/GED",5: "cllg,y1-3",6: "cllg,y4+"},
            'age_cat5':{1:'18-24',2:'25-29',3:'30-34',4:'35-39',5:'40-44',6:'45-49',7:'50-54',8:'55-59',9:'60-64',10:'65-69',11:'70-74',12:'75-79',13:'80+'}
    }
    default_good = {}
    default_good = {'sex':2,
            'cholestrol': 2, 
            'blood_pressure':3,
            "diabetes":3,
            'exer':2,
            'diff_walk':2,
            'marital_status': 5,
            'employ_status': 1,
            'BMI5':2500,
            'BMI_cat':2,
            'income':8,
            'mental_poor30':0,
            'phys_ment_poor30':0,
            'smoke':3,
            'CHILDREN':0,
            'checkup':1,
            'alcohol30':0,
            'EDUCA':6,
            'age_cat5':1
    }
    default_bad = {}
    default_bad = {'sex':1,
            'cholestrol': 1, 
            'blood_pressure':1,
            "diabetes":1,
            'exer':2,
            'diff_walk':2,
            'marital_status': 3,
            'employ_status': 3,
            'BMI5':1800,
            'BMI_cat':1,
            'income':1,
            'mental_poor30':30,
            'phys_ment_poor30':30,
            'smoke':1,
            'CHILDREN':12,
            'checkup':4,
            'alcohol30':30,
            'EDUCA':2,
            'age_cat5':5
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

    return quest, desc,known_binary, AXlabel,default_good,default_bad

'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
# If in app.py "Sim.Create_EDAplot" is defined as True, 
this function creates the EDA plots
'''
def Create_EDA_Plots():
    quest, desc,known_binary,AXlabel,default_good,default_bad = variable_desc()

    feature_convert_name_for_vis = ['age_cat5','EDUCA','BMI_cat','employ_status','marital_status','cholestrol','blood_pressure']

    feature_eda = feature_convert_name_for_vis  + ['alcohol30','smoke']
    y = ['attack','stroke',"other_cancer"]#,"skin_cancer"

    col = len(y)
    row = len(feature_convert_name_for_vis)
    width = col*7
    height = row*7

    fig, ax = plt.subplots(row,col,figsize = (width,height))
    plt.subplots_adjust(wspace=.3, hspace=0.6)

    sns.set(style="whitegrid")
    ax[0][0].set_title('Attack', fontsize=20)
    ax[0][1].set_title('Stroke', fontsize=20)
    ax[0][2].set_title('Cancer', fontsize=20)


    for i_idx,i in enumerate(feature_convert_name_for_vis):
        x=list(desc[i].keys())
        xtick_labels = desc[i]

        for j_idx,j in enumerate(y):
            ax[i_idx][j_idx].set_xticks(x)

            labels = [xtick_labels.get(i, '') for i in x]
            sns.barplot(x=i,y=j,hue='sex', data = df_dropna, ax = ax[i_idx][j_idx])

            ax[i_idx][j_idx].set_xticklabels(labels, rotation=90, ha='left', x=-0.5,fontsize=15)  # Adjust the rotation angle as needed
            ax[i_idx][j_idx].set_ylabel(j, fontsize=15)
            ax[i_idx][j_idx].set_xlabel(AXlabel[i],fontsize=15)
            ax[i_idx][j_idx].legend(fontsize=18) 
    plt.tight_layout()                
    plt.savefig('saved/Figs/EDA_Lifestyle.png',dpi = 300)
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

    df_dropna = pd.read_csv("data/Lifestyle2021_Clean.csv")

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

    # setting.df_dropna_fe = df_dropna_fe
    # setting.variables_to_process = variables_to_process
    # setting.feature_names = feature_names
    return preprocessor, df_dropna_fe,variables_to_process,feature_names

'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
The function that take the information from the user for learning features and then apply 
the feature engineering (based on the feature importance used for learning)
'''
def func_input_user(preprocessor,preprocessor_nn, variables_to_process,feature_names):
    ordinal_variables_test = ['sex','cholestrol','blood_pressure','EDUCA','income','age_cat5','diff_walk','marital_status','employ_status','exer','smoke','checkup','diabetes']

    ordinal30 = ['alcohol30','mental_poor30']#,'phys_ment_poor30'

    questions, desc, known_binary,AXlabel,default_good,default_bad = variable_desc()

    st.sidebar.write("**Interactive Element**")

    user_input2 = {}

    i = 'BMI5'
    user_input2[i] = st.sidebar.slider(f"{questions[i]}:\n{desc[i]}",1000,6000,1800)

    for i in ordinal_variables_test:
        rang = list(desc[i].keys())
        user_input2[i] = st.sidebar.selectbox(f"{questions[i]}:\n{desc[i]}", rang, index=rang.index(default_bad[i]))


    for i in ordinal30:
        user_input2[i] = st.sidebar.selectbox(f"{questions[i]}:\n{desc[i]}", range(1,30))


    df_input = pd.DataFrame(user_input2,index = ['user_input'])
    df_input = df_input[variables_to_process]
    df_input = df_input.astype(int).dropna().replace(known_binary)

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
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=20)

    class_counts = y.value_counts()

    # Identify the minority class
    minority_class = class_counts.idxmin()

    # Count of instances in the minority class
    minority_class_count = class_counts[minority_class]
    rus = RandomUnderSampler(sampling_strategy={0:minority_class_count}, random_state=10) # reducing to 3016
    X_rus, y_rus = rus.fit_resample(X, y)


    X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus, test_size=0.3, random_state=20)

    return X_train, X_test, y_train, y_test


