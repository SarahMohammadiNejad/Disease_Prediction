import streamlit as st
import pandas as pd

from sklearn.linear_model import LogisticRegression
from funcs import func_calculate_metrics
import pickle
'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
A function for defining and fitting logistic regression
and metrics and confusion matrix. It also caluculates 
feature importance and save it as csv file. It also saves 
the model for later use if we don't want to not train the model
'''
@st.cache_resource
def LogReg(X_train, y_train, X_test, y_test,out,dataset):

    lr = LogisticRegression().fit(X_train, y_train)

    coeff_labels = ['lr']
    coeff_models = [lr]

    metrics, cm = func_calculate_metrics(X_train, y_train, X_test, y_test,coeff_labels, coeff_models)

    coefficients = lr.coef_[0]
    feature_importance_lr = pd.DataFrame({'Feature': X_train.columns, 'importance': coefficients,'method': 'lr'})

    path2 = f'saved/metrics_{dataset}/{out}'
    feature_importance_lr.to_csv(path2+'_lr_featureImp.csv')

    path = f'saved/saved_models_{dataset}'
    filename = f'{path}/trained_lr_{out}.pkl'
    with open(filename, 'wb') as model_file:
        pickle.dump(lr, model_file)

    return lr,metrics,cm