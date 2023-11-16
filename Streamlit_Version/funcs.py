import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
A function that calculates all the scores
'''
def measure_error(y_true, y_pred, label):
    num_dig = 3
    return pd.Series({'accu.':round(accuracy_score(y_true, y_pred),num_dig),
                      'prec.': round(precision_score(y_true, y_pred),num_dig),
                      'recall': round(recall_score(y_true, y_pred),num_dig),
                      'f1': round(f1_score(y_true, y_pred),num_dig)},
                      name=label)

'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
A function to gather the scores and confusion matrix in 
train and test sets in all models except Neural Network model
'''
def func_calculate_metrics(X_train, y_train, X_test, y_test,coeff_labels, coeff_models):
    y_pred = list()
    y_pred_train = list()
    metrics = list()

    for lab, mod in zip(coeff_labels, coeff_models):

        y_pred_now = pd.Series(mod.predict(X_train), name=lab)
        y_pred.append(y_pred_now)

        # The error metrics
        metrics.append(measure_error(y_train, y_pred_now, label=f'{lab}_train'))
        y_pred_now = pd.Series(mod.predict(X_test), name=lab)
        y_pred.append(y_pred_now)

        # The error metrics
        metrics.append(measure_error(y_test, y_pred_now, label=f'{lab}_test'))

        # The confusion matrix
        cm = confusion_matrix(y_test, y_pred_now)

    metrics = pd.concat(metrics, axis=1)
    return metrics, cm 
'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
A function to gather the scores and confusion matrix in train and test sets in Neural Network model
'''
def func_calculate_metrics_nn(X_train, y_train, X_test, y_test,coeff_labels, coeff_models):
    y_pred = list()
    y_pred_train = list()
    metrics = list()

    for lab, mod in zip(coeff_labels, coeff_models):

        y_pred = pd.DataFrame((mod.predict(X_train) > 0.5).astype("int32"))

        # The error metrics
        metrics.append(measure_error(y_train, y_pred, label=f'{lab}_train'))
        y_pred = pd.DataFrame((mod.predict(X_test) > 0.5).astype("int32"))

        # The error metrics
        metrics.append(measure_error(y_test, y_pred, label=f'{lab}_test'))

        # The confusion matrix
        cm = confusion_matrix(y_test, y_pred)

    metrics = pd.concat(metrics, axis=1)

    return metrics, cm 
'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
A function to plot confusion matrix 
'''    
def plot_conf_mat(nrows,ncols,coeff_labels,cm):
    fig, axList = plt.subplots(nrows, ncols)

    axList = axList.flatten()
    len = ncols*2.5
    hei = nrows * 2.5
    fig.set_size_inches(len, hei)

    for ax,lab in zip(axList, coeff_labels):
        sns.heatmap(cm[lab], ax=ax, annot=True, fmt='d',
                    annot_kws={"size": 13, "weight": "bold"}, cbar=False)
    
        ax.set(title=lab)

        labels = ['No', 'Yes']
        ax.set_xticklabels(labels, fontsize=11);
        ax.set_yticklabels(labels, fontsize=11);
        ax.set_ylabel('Truth', fontsize=12);
        ax.set_xlabel('Pred.', fontsize=12)
        
    plt.tight_layout()
    st.pyplot(fig)

'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
# A function to create the end text
'''
def Create_Text_End():
    st.markdown("#### The Advantage of the Code")
    st.markdown("- The code is automatic and you can easily apply it to any new dataset just after cleaning. ")
    st.markdown("- After cleaning data, just group your features to categoical/numerical variables. Balancing the data, Feature engineering for the data and user input, Optimizing the models are done automatically.")
    st.markdown("- To save time, there is the option to save ML models and read the next time as long as we are not going to change the parameters. ")
    st.markdown("#### To Do in Future:")
    st.markdown("- To deploy the code as a web application.")
    st.markdown("- To extend the model and add more disease.")
    st.markdown("- To add some features that we have control on in our daily life.")
    st.markdown("- To calculate p_value and null hypothesis.")
    st.markdown("- To make suggestion to the user what to  change in daily life.")
'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
# A function to write about learning
'''
def Create_Text_Learning():
    st.markdown("- Logistic Regression (LR),")
    st.markdown("- Support Vector Classifier with Linear kernel (SVCL): GridSearchCV for C,")
    st.markdown("- Support Vector Classifier with Gaussian kernel (SVCG):  GridSearchCV for gamma,")
    st.markdown("- Decision Tree (DT): GridSearchCV for max_depth, and max_features,")
    st.markdown("- Random Forest (RF): GridSearchCV for n_estimators, max_depth, and max_features,")
    st.markdown("- Neural Network (NN):   \n Dense1: 18 neurons, activation = sigmoid,  \n Dense2: 1 neurons, activation = sigmoid   \n nn.compile(Optimizer = Adam(), loss = 'binary_crossentropy', metrics=['accuracy']).")
    st.markdown('**Gridsearch done separately for each disease**')

'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
# A function to write Introduction text
'''
def Create_Text_Intro():
    st.title("Predicting Diseases; Clinical and Lifestyle")
    st.image("data/Fig_Body.jpg", caption=" ", use_column_width=True)

    st.markdown("### Introduction")
    st.write("Every year millions of people die from cardiovascular disease(CVDs) and different types of cancer around the world. Some of the most common risk factors are non-healthy lifestyle, obesity, high blood pressure, high cholesterol, long-term stress, diabetes. One can prevent disease by having a healthy lifestyle: by being physically active, choosing healthy diet, avoiding smoking, managing stress, and getting regular health screens.")


    st.markdown("##### Main objectives")
    st.markdown("- Creating machine learning models and the best hyperparameters to classify people for each disease.")
    st.markdown("- Finding the features with highest impact on each disease.")

    st.markdown("##### 1. Clinical Dataset")
    st.write(" UCI repository link: https://archive.ics.uci.edu/ml/datasets/Heart+Disease. \n 303 samples, Target: heart disease, 14 featurers: mostly clinical measurements:  ")
    st.write("**categorical_variables:**  sex, chest_pain_type, rest_ecg, st_slope, thalassemia, fasting_blood_suger, exercise_induced_angina")
    st.write("**numerical_variables:** age, num_major_vessels, max_heart_rate_achieved, cholesterol, st_depression, resting_blood_pressure")
    # st.write("I don't know of many of the parameters myself")
    # st.markdown("- age: Age in years")
    # st.markdown("- sex: The person's sex (1: male, 0: female)")
    # st.markdown("- cp: The chest pain type experienced by the person (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic) ")
    # st.markdown("- trestbps: Resting blood pressure (mm Hg on admission to the hospital)")
    # st.markdown("- chol: serum cholestoral in mg/dl")
    # st.markdown("- fbs: Fasting blood sugar > 120 mg/dl (1: true, 0: false)")
    # st.markdown("- restecg: Resting electrocardiographic measurement (0: normal, 1 = having ST-T wave abnormality, 2: showing probable or definite left ventricular hypertrophy by Estes' criteria)")
    # st.markdown("- thalach: Maximum heart rate achieved")
    # st.markdown("- exang: Exercise induced chest pain (1: yes, 0: no)")
    st.markdown("- st_slope: ECG readout indicates quality of blood flow.  \n - thalassemia: Blood disorder, shows blood flow to the heart   \n - number of major vessels colored by flouroscopy   \n - st_depression induced by exercise. A measure of abnormality in ECG.")# to the heart (upsloping, flat, downsloping)
    # st.markdown("- thalassemia: Blood disorder, shows blood flow to the heart")#results of thallium stress test,  (normal, fixed defect, reversable defect)
    # st.markdown("- number of major vessels colored by flouroscopy")
    # st.markdown("- st_depression induced by exercise. A measure of abnormality in ECG.")
    st.write('More interesting for clinical purposes, for people who are already diagnosed to have one serious risk factor.')

    st.markdown("##### 2. Lifestyle Dataset")
    st.write('Centers for Disease Control and Prevention: https://www.cdc.gov/brfss/annual_data/annual_2022.html, (a survey, every year).  \n Large sample size (450,000), Target: CVD dieseases as well as different type of cancers, 300 features. ')
    st.write("**Target disease:** attack, stroke, angina, skin cancer, other cancer")
    st.write("**Categorical_variables:** sex, cholesterol, blood pressure, exercise, difficulty walking, marital status, employment status, diabetes.")
    st.write("**Ordinal variables:** age, education, BMI, income, smoke, mental health, alcohol, checkup.")
    #  The good point is that you can choose between fetures as you want, it has data for heart diesease and cancer and you  can even choose between different cancers. And this makes it more interesting because it has interesting informations but at the same time you need to first read the codebook first carefuly and choose the features yourself. 

    st.write('**Note**: Dataset is imbalanced toward the healthy group. RandomUnderSampler, minority_class_count')
    st.write('\t\t\t\t\t\t\t\t\t------------------------------')


