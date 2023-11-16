import streamlit as st
import pandas as pd
import numpy as np
import pickle

from LogisticReg import LogReg
from SupportVector import svcl,svcg,calibrated_svcl,calibrated_svcg
from RandomForest import rf,dt
from NeuralNetwork import NeurNet
'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
'''
def Initialize_Models(Sim):

    # for out in dependent_variables[:1]:
    path = f'saved/saved_models_{Sim.dataset}'
    mod = {}
    models = {}

    for ml in Sim.saved_model_labels:
        mod = {}
        for out in Sim.dependent_variables[:Sim.num_disease]:
        
            filename = f'{path}/trained_{ml}_{out}.pkl'
            with open(filename, 'rb') as model_file:
                mod[out] = pickle.load(model_file)
        models[ml]=mod

    return models

'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
This function gathers all the models that are going to be trained and 
concat all their metrics and confusion matrix together and save them 
as csv and npz files to be presented later.
'''

def Learning_and_Metrics(Sim,df_dropna_fe,df_dropna_fe_nn):
    model_lr = {}
    model_rf = {}
    model_dt = {}
    model_svcl = {}
    model_calibrated_svcl = {}
    model_svcg = {}
    model_calibrated_svcg = {}
    model_nn = {}

    models = {}
    all_metrics = pd.DataFrame()

    if(Sim.dataset=='Clinical'):
        from DataProcess_clinical import TrainTest
    elif(Sim.dataset=='Lifestyle'):
        from DataProcess_Lifestyle import TrainTest

    for out in Sim.dependent_variables[0:Sim.num_disease]:
        all_metrics_disease = pd.DataFrame()

    # ----------------------
        X_train, X_test, y_train, y_test = TrainTest(Sim,df_dropna_fe,out)
        X_train_nn, X_test_nn, y_train_nn, y_test_nn = TrainTest(Sim,df_dropna_fe_nn,out)

        cms = {}

        st.write(f'\t\t\t\tLearning for disease: {out} ...')

        if Sim.DoTrai['lr']:
            model_lr[out],metrics_lr, cms['lr'] = LogReg(X_train, y_train, X_test, y_test,out,Sim.dataset)
            all_metrics_disease = pd.concat([all_metrics_disease,metrics_lr],axis = 1)#metrics_svcl,metrics_svcg,metrics_rf

        if Sim.DoTrai['svcl0']:
            model_svcl[out],metrics_svcl,_ = svcl(X_train, y_train, X_test, y_test,out,Sim.dataset)  

        if Sim.DoTrai['svcg0']:
            model_svcg[out],metrics_svcg,_ = svcg(X_train, y_train, X_test, y_test,out,Sim.dataset)  

        if Sim.DoTrai['svcl']:
            model_calibrated_svcl[out],metrics_calibrated_svcl, cms['svcl'] = calibrated_svcl(X_train, y_train, X_test, y_test,out,Sim.dataset)         
            all_metrics_disease = pd.concat([all_metrics_disease,metrics_calibrated_svcl],axis = 1)#metrics_svcl,metrics_svcg,metrics_rf

        if Sim.DoTrai['svcg']:
            model_calibrated_svcg[out],metrics_calibrated_svcg, cms['svcg'] = calibrated_svcg(X_train, y_train, X_test, y_test,out,Sim.dataset)
            all_metrics_disease = pd.concat([all_metrics_disease,metrics_calibrated_svcg],axis = 1)#metrics_svcl,metrics_svcg,metrics_rf

        if Sim.DoTrai['dt']:
            model_dt[out],metrics_dt, cms['dt'] = dt(X_train, y_train, X_test, y_test,out,Sim.dataset)
            all_metrics_disease = pd.concat([all_metrics_disease,metrics_dt],axis = 1)#metrics_svcl,metrics_svcg,metrics_rf

        if Sim.DoTrai['rf']:
            model_rf[out],metrics_rf, cms['rf'] = rf(X_train, y_train, X_test, y_test,out,Sim.dataset)
            all_metrics_disease = pd.concat([all_metrics_disease,metrics_rf],axis = 1)#metrics_svcl,metrics_svcg,metrics_rf

        if Sim.DoTrai['nn']:
            model_nn[out], metrics_nn, cms['nn'] = NeurNet(X_train_nn, y_train_nn, X_test_nn, y_test_nn,out,Sim.dataset)
            all_metrics_disease = pd.concat([all_metrics_disease,metrics_nn],axis = 1)#metrics_svcl,metrics_svcg,metrics_rf


        path2 = f'saved/metrics_{Sim.dataset}/{out}_'
        all_metrics_disease.to_csv(path2+'metrics.csv')

        np.savez(path2+'ConfusionM.npz', **cms)

    models['lr'] = model_lr  
    models['svcl0'] = model_svcl   
    models['svcg0'] = model_svcg  
    models['svcl'] = model_calibrated_svcl   
    models['svcg'] = model_calibrated_svcg 
    models['rf'] = model_rf 
    models['dt'] = model_dt  
    models['nn'] = model_nn   
    return models

'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
This function predict the probability that the user will have the disease in future. 
Before this, the "func_input_user" function take the information from the user for 
learning features and apply the feature iportance on it. 
'''
def predict_userInput(Sim,df_input_fe,df_input_fe_nn):

    all_prob = pd.DataFrame()

    num_dec = 3
    df_prob = pd.DataFrame()
    df_predict = pd.DataFrame()
    for out in Sim.dependent_variables[:Sim.num_disease]:

        prob = {}
        predict = {}
        df_prob_temp = pd.DataFrame()
        df_predict_temp = pd.DataFrame()

        label_prob = f'Prob. {out}'
        label_pred = f"Will have {out}?"
        for ml in list(Sim.models2predict)[:-1]:

            prob[ml] = np.round(Sim.models2predict[ml][out].predict_proba(df_input_fe)[0],num_dec)
            predict[ml] = Sim.models2predict[ml][out].predict(df_input_fe)


        test0 = Sim.models2predict['nn'][out].predict(df_input_fe_nn)
        test1 = 1-test0[0]
        t0 = np.round(test0[0],num_dec)
        t1 = np.round(test1,num_dec)

        t00 = t0[0]
        t10 = t1[0]

        tt = np.array([t10, t00])#.reshape([2,])
        prob['nn'] = tt
        test3 = (Sim.models2predict['nn'][out].predict(df_input_fe_nn) > 0.5).astype(int)#"int32"
        ttt = test3[0].reshape([1,])
        predict['nn'] = ttt


        df_prob_temp = pd.DataFrame(data = prob, index = ['Prob. healthy',label_prob])
        df_prob_temp_round = df_prob_temp#.round(3)

        df_predict_temp = pd.DataFrame(data = predict,index = [label_pred])
        df_predict_temp.replace(0,'No', inplace= True)
        df_predict_temp.replace(1,'Yes', inplace= True)
        
        df_prob = pd.concat([df_prob_temp_round,df_predict_temp],axis = 0)
        if Sim.DoPredInpu['nn']:
            df_prob['nn']['Prob. healthy'] = round(df_prob['nn']['Prob. healthy'],3)#.round(3)
            df_prob['nn'][label_prob] = round(df_prob['nn'][label_prob],3)#.round(3)

        all_prob = pd.concat([all_prob,df_prob],axis = 0)
    if Sim.dataset == 'Lifestyle':
        st.write(all_prob.iloc[[1,4,7,10,13]])
        # st.write(all_prob)

    else:
        st.write(df_prob)


