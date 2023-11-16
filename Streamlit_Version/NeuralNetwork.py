import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from keras.models  import Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
import tensorflow.keras as keras

from funcs import func_calculate_metrics_nn
'''
*************************************************
*               A NEW FUNCTION                  *
*************************************************
A function for defining and fitting Neural Network
and metrics and confusion matrix. It also caluculates 
feature importance and save it as csv file. It also saves 
the model for later use if we don't want to not train the model
'''
@st.cache_resource
def NeurNet(X_train, y_train, X_test, y_test,out,dataset):
    if dataset == 'Clinical':
        N_epoch = 500
    else:
        N_epoch = 100

    nn = Sequential([
        Dense(18,input_shape = (X_train.shape[1],),activation = 'sigmoid'),
        Dense(1,activation='sigmoid')
    ])
    from keras.optimizers import Adam, SGD, RMSprop
    nn.compile(Adam(), loss = "binary_crossentropy", metrics=["accuracy"])

    nn_hist = nn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=N_epoch)#,callbacks = [early_stop])
    coeff_labels = ['nn']
    coeff_models = [nn]

    path2 = f'saved/metrics_{dataset}/{out}_'
    nn_history = pd.DataFrame(nn_hist.history)
    nn_history.to_csv(path2+'nn_hist.csv')
    metrics, cm = func_calculate_metrics_nn(X_train, y_train, X_test, y_test,coeff_labels, coeff_models)

    path = f'saved/saved_models_{dataset}'

    import pickle
    filename = f'{path}/trained_nn_{out}.pkl'
    with open(filename, 'wb') as model_file:
        pickle.dump(nn, model_file)
    return nn,metrics,cm