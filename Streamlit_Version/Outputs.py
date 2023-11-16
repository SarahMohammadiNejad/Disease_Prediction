import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

from funcs import plot_conf_mat

'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
'''
def Plots_Output(out,Sim):
    path = f'saved/metrics_{Sim.dataset}/{out}_'

    if(Sim.DoPlot_ConfusionM):
        st.write(f'Confusion Matrix for {out}')        
        cms = np.load(path+'ConfusionM.npz')
        st.write(path+'ConfusionM.npz')
        plot_conf_mat(2,3, list(Sim.models2showResult),cms)


    if(Sim.DoWrite_metrics):
        st.write(f'Metrics for {out}')
        all_metrics = pd.read_csv(path+'metrics.csv', index_col= 0)

        metric_list2show_train = ['lr_train','svcl_train','svcg_train','dt_train','rf_train','nn_train']
        all_metrics_train = all_metrics[metric_list2show_train]
        all_metrics_train.rename(columns={'lr_train': 'LR', 'svcl_train': 'SVCL','svcg_train':'SVCG','dt_train':'DT','rf_train':'RF','nn_train':'NN'}, inplace=True)

        st.write('metrics for train set')
        st.write(all_metrics_train)
        metric_list2show_test = ['lr_test','svcl_test','svcg_test','dt_test','rf_test','nn_test']
        all_metrics_test = all_metrics[metric_list2show_test]
        all_metrics_test.rename(columns={'lr_test': 'LR', 'svcl_test': 'SVCL','svcg_test':'SVCG','dt_test':'DT','rf_test':'RF','nn_test':'NN'}, inplace=True)

        st.write('metrics for test set')
        st.write(all_metrics_test)


    if Sim.DoPlot_NNloss:
        nn_hist = pd.read_csv(path+'nn_hist.csv', index_col= 0)
        n = nn_hist["loss"].shape[0]

        fig = plt.figure(figsize=(9, 4.5))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(range(n), (nn_hist["loss"]),'r.', label="Train Loss")
        ax.plot(range(n), (nn_hist["val_loss"]),'b.', label="Validation Loss")
        # ax.plot(range(xlim), (nn_hist["loss"][:xlim]),'r.', label="Train Loss")
        # ax.plot(range(xlim), (nn_hist["val_loss"][:xlim]),'b.', label="Validation Loss")
        ax.legend()
        ax.set_title('Loss over iterations')

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(range(n), (nn_hist["accuracy"]),'r.', label="Train Acc")
        ax.plot(range(n), (nn_hist["val_accuracy"]),'b.', label="Validation Acc")

        ax.legend(loc='lower right')
        ax.set_title('Accuracy over iterations')
        # plt.savefig('SGD_1_Loss.pdf')
        st.pyplot(fig)

    if Sim.DoPlot_FeatureImp:
        # show_FeatureImp = ['lr','svcl0','svcg0','rf']
        show_FeatureImp = ['svcg0']

        # fig = plt.figure(figsize=(9, 4.5))
        for ii_idx,ii in enumerate(show_FeatureImp):
            # ax = fig.add_subplot(2, 2, ii_idx+1)
            fi_uns = pd.read_csv(path+ii+'_featureImp.csv', index_col= 0)  
            fi_uns['imp_abs'] = np.abs(fi_uns['importance'])
            fi_uns['imp_sgn'] = np.sign(fi_uns['importance'])

            fi= fi_uns.sort_values(by = ['imp_abs'], ascending=False)
            fig, ax = plt.subplots()
            # fi.plot.barh(x='Feature', y='importance', figsize=(10, 15),ax=ax) 
            fi.plot.bar(x='Feature', y='importance',figsize=(10, 5),ax=ax) 

            st.pyplot(fig)