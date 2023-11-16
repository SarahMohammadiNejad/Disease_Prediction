import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import pickle

from funcs import func_calculate_metrics

gamma_list = [0.05, 0.1, 0.2,.5, 1, 2]
C_list = np.arange(0.2,0.9,0.1)
'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
A function for defining and fitting Calibrated Support Vector with linear Kernel
and metrics and confusion matrix. It also caluculates 
feature importance and save it as csv file. It also saves 
the model for later use if we don't want to not train the model
'''
@st.cache_resource
def calibrated_svcl(X_train, y_train, X_test, y_test,out,dataset):

    param_svcl_grid = {'C':C_list}
    svcl_grid = GridSearchCV(SVC(kernel = "linear"),
                    param_grid=param_svcl_grid,
                    scoring='accuracy',
                    cv=5,
                    n_jobs=1)
    svcl_grid = svcl_grid.fit(X_train, y_train)

    # svcl_grid.cv_results_
    column_list_svcl = [
                'param_C',
                'mean_test_score',
                'std_test_score',
                'rank_test_score'
                ]

    # create result dataframe
    result_svcl = pd.DataFrame(svcl_grid.cv_results_)[column_list_svcl]
    result_svcl.sort_values(by = ['rank_test_score'],inplace = True)
    result_svcl.reset_index(inplace = True)

    svcl= SVC(probability=True,kernel = "linear", C = result_svcl.loc[0]['param_C'])
    calibrated_svcl = CalibratedClassifierCV(base_estimator=svcl, method='sigmoid')

    calibrated_svcl.fit(X_train, y_train)

    coeff_labels = ['svcl']
    coeff_models = [calibrated_svcl]

    metrics, cm = func_calculate_metrics(X_train, y_train, X_test, y_test,coeff_labels, coeff_models)

    path = f'saved/saved_models_{dataset}'

    import pickle
    filename = f'{path}/trained_svcl_{out}.pkl'

    with open(filename, 'wb') as model_file:
        pickle.dump(calibrated_svcl, model_file)

    return calibrated_svcl,metrics,cm

'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
A function for defining and fitting Calibrated Support Vector 
with gaussian Kernel and metrics and confusion matrix. It also caluculates 
feature importance and save it as csv file. It also saves 
the model for later use if we don't want to not train the model
'''
@st.cache_resource
def calibrated_svcg(X_train, y_train, X_test, y_test,out,dataset):

    param_svcg_grid = {'gamma':gamma_list}
    svcg_grid = GridSearchCV(SVC(kernel='rbf'),
                    param_grid=param_svcg_grid,
                    scoring='accuracy',
                    cv=5,
                    n_jobs=1)
    svcg_grid = svcg_grid.fit(X_train, y_train)

    column_list_svcg = [
                'param_gamma',
                'mean_test_score',
                'std_test_score',
                'rank_test_score'
                ]

    # create result dataframe
    result_svcg = pd.DataFrame(svcg_grid.cv_results_)[column_list_svcg]
    result_svcg.sort_values(by = ['rank_test_score'],inplace = True)
    result_svcg.reset_index(inplace = True)

    svcg= SVC(probability=True,kernel='rbf', gamma = result_svcg.loc[0]['param_gamma'])
    calibrated_svcg = CalibratedClassifierCV(base_estimator=svcg, method='sigmoid')

    calibrated_svcg.fit(X_train, y_train)

    coeff_labels = ['svcg']
    coeff_models = [calibrated_svcg]


    metrics, cm = func_calculate_metrics(X_train, y_train, X_test, y_test,coeff_labels, coeff_models)

    path = f'saved/saved_models_{dataset}'
    import pickle
    filename = f'{path}/trained_svcg_{out}.pkl'
    with open(filename, 'wb') as model_file:
        pickle.dump(calibrated_svcg, model_file)

    return calibrated_svcg,metrics,cm

'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
A function for defining and fitting Support Vector with linear Kernel
and metrics and confusion matrix. It also caluculates 
feature importance and save it as csv file. It also saves 
the model for later use if we don't want to not train the model
'''
@st.cache_resource
def svcl(X_train, y_train, X_test, y_test,out,dataset):

    param_svcl_grid = {'C':C_list}
    svcl_grid = GridSearchCV(SVC(kernel = "linear"),
                    param_grid=param_svcl_grid,
                    scoring='accuracy',
                    cv=5,
                    n_jobs=1)
    svcl_grid = svcl_grid.fit(X_train, y_train)

    column_list_svcl = [
                'param_C',
                'mean_test_score',
                'std_test_score',
                'rank_test_score'
                ]

    # create result dataframe
    result_svcl = pd.DataFrame(svcl_grid.cv_results_)[column_list_svcl]
    result_svcl.sort_values(by = ['rank_test_score'],inplace = True)
    result_svcl.reset_index(inplace = True)

    svcl= SVC(kernel = "linear", C = result_svcl.loc[0]['param_C'])

    svcl.fit(X_train, y_train)

    coeff_labels = ['svcl0']
    coeff_models = [svcl]

    metrics, cm = func_calculate_metrics(X_train, y_train, X_test, y_test,coeff_labels, coeff_models)

    path = f'saved/saved_models_{dataset}'

    filename = f'{path}/trained_svcl0_{out}.pkl'

    with open(filename, 'wb') as model_file:
        pickle.dump(svcl, model_file)

    coefficients = svcl.coef_[0]
    feature_importance_svcl = pd.DataFrame({'Feature': X_train.columns, 'importance': coefficients,'method': 'svcl'})

    path2 = f'saved/metrics_{dataset}/{out}'
    feature_importance_svcl.to_csv(path2+'_svcl0_featureImp.csv')


    return svcl,metrics,cm

'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
A function for defining and fitting Support Vector with gaussian Kernel
and metrics and confusion matrix. It also caluculates 
feature importance and save it as csv file. It also saves 
the model for later use if we don't want to not train the model
'''
@st.cache_resource
def svcg(X_train, y_train, X_test, y_test,out,dataset):

    param_svcg_grid = {'gamma':gamma_list}
    svcg_grid = GridSearchCV(SVC(kernel='rbf'),
                    param_grid=param_svcg_grid,
                    scoring='accuracy',
                    cv=5,
                    n_jobs=1)
    svcg_grid = svcg_grid.fit(X_train, y_train)

    column_list_svcg = [
                'param_gamma',
                'mean_test_score',
                'std_test_score',
                'rank_test_score'
                ]

    # create result dataframe
    result_svcg = pd.DataFrame(svcg_grid.cv_results_)[column_list_svcg]
    result_svcg.sort_values(by = ['rank_test_score'],inplace = True)
    result_svcg.reset_index(inplace = True)

    svcg= SVC(kernel='rbf', gamma = result_svcg.loc[0]['param_gamma'])
    svcg.fit(X_train, y_train)

    coeff_labels = ['svcg0']
    coeff_models = [svcg]


    metrics, cm = func_calculate_metrics(X_train, y_train, X_test, y_test,coeff_labels, coeff_models)

    support_vector_indices = svcg.support_
    support_vector_coefficients = svcg.dual_coef_.ravel()
    feature_importance_scores = support_vector_coefficients @ svcg.support_vectors_
    feature_importance_svcg = pd.DataFrame({'Feature': X_train.columns, 'importance': feature_importance_scores,'method': 'svcg'})
    path2 = f'saved/metrics_{dataset}/{out}'
    feature_importance_svcg.to_csv(path2+'_svcg0_featureImp.csv')


    path = f'saved/saved_models_{dataset}'

    import pickle
    filename = f'{path}/trained_svcg0_{out}.pkl'

    with open(filename, 'wb') as model_file:
        pickle.dump(svcg, model_file)

    return svcg,metrics,cm
