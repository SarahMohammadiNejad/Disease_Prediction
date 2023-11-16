import streamlit as st
import pandas as pd
import numpy as np
# import seaborn as sns
# import plotly.express as px
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle

from funcs import func_calculate_metrics

'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
A function for defining and fitting Rando Forest
and metrics and confusion matrix. It also caluculates 
feature importance and save it as csv file. It also saves 
the model for later use if we don't want to not train the model
'''
num_estimator = [30,40, 50,70, 100] #40, 50,70, 
max_feat_range = range(5, 20,2)
max_depth_range = range(1,6)

@st.cache_resource
def rf(X_train, y_train, X_test, y_test,out,dataset):
    param_grid = {'n_estimators':num_estimator,
                'max_depth':max_depth_range,
                'max_features': max_feat_range}

    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    param_grid=param_grid,
                    scoring='accuracy',
                    cv = 5,
                    n_jobs=1)

    rf_grid = rf_grid.fit(X_train, y_train)

    column_list_rf = [
                'param_n_estimators',  
                'param_max_depth',
                'param_max_features',
                'mean_test_score',
                'std_test_score',
                'rank_test_score'
                ]

    # create result dataframe
    result_rf = pd.DataFrame(rf_grid.cv_results_)[column_list_rf]
    result_rf.sort_values(by = ['rank_test_score'],inplace = True)
    result_rf.reset_index(inplace = True)

    rf = RandomForestClassifier(random_state=42,
                                max_depth=result_rf.loc[0]['param_max_depth'], max_features=result_rf.loc[0]['param_max_features'],
                            n_estimators=result_rf.loc[0]['param_n_estimators']).fit(X_train, y_train)

    coeff_labels = ['rf']
    coeff_models = [rf]

    metrics, cm = func_calculate_metrics(X_train, y_train, X_test, y_test,coeff_labels, coeff_models)

    feature_importance_rf = pd.DataFrame({'Feature': X_train.columns, 'importance': rf.feature_importances_,'method': 'rf'})
    path2 = f'saved/metrics_{dataset}/{out}_'
    feature_importance_rf.to_csv(path2+'rf_featureImp.csv') 


    path = f'saved/saved_models_{dataset}'

    filename = f'{path}/trained_rf_{out}.pkl'
    with open(filename, 'wb') as model_file:
        pickle.dump(rf, model_file)

    return rf,metrics,cm

'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
A function for defining and fitting Decision Tree
and metrics and confusion matrix. It also caluculates 
feature importance and save it as csv file. It also saves 
the model for later use if we don't want to not train the model
'''
from sklearn.tree import DecisionTreeClassifier
@st.cache_resource
def dt(X_train, y_train, X_test, y_test,out,dataset):
    # st.write(X_train.columns)
    param_grid = {'max_depth':max_depth_range,
                'max_features': max_feat_range}

    dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                    param_grid=param_grid,
                    scoring='accuracy',
                    cv = 5,
                    n_jobs=1)

    dt_grid = dt_grid.fit(X_train, y_train)

    column_list_dt = [
                'param_max_depth',
                'param_max_features',
                'mean_test_score',
                'std_test_score',
                'rank_test_score'
                ]

    result_dt = pd.DataFrame(dt_grid.cv_results_)[column_list_dt]
    result_dt.sort_values(by = ['rank_test_score'],inplace = True)
    result_dt.reset_index(inplace = True)

    dt = DecisionTreeClassifier(max_depth = result_dt.loc[0]['param_max_depth'], max_features = result_dt.loc[0]['param_max_features'],random_state=2).fit(X_train, y_train)
    coeff_labels = ['dt']
    coeff_models = [dt]

    metrics, cm = func_calculate_metrics(X_train, y_train, X_test, y_test,coeff_labels, coeff_models)

    path = f'saved/saved_models_{dataset}'
    filename = f'{path}/trained_dt_{out}.pkl'
    with open(filename, 'wb') as model_file:
        pickle.dump(dt, model_file)

    return dt,metrics,cm

