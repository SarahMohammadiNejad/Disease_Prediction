import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from funcs import plot_conf_mat
from funcs import Create_Text_Intro, Create_Text_Learning,Create_Text_End
from MachineLearning import Initialize_Models,Learning_and_Metrics,predict_userInput
from Outputs import Plots_Output

from simulation_class import Simulation

st.set_page_config(layout="wide")

#================================
# In the left sidebar, user select between the two dataframe to study disease!
dataset = st.sidebar.selectbox("Select Dataset", ['Clinical','Lifestyle'])
Sim = Simulation(dataset)

Create_Text_Intro() # A function to write Introduction text

if(Sim.dataset=='Clinical'):
    from DataProcess_clinical import func_read_fe,variable_desc,variable_grouping,Create_EDA_Plots,func_input_user
elif(Sim.dataset=='Lifestyle'):
    from DataProcess_Lifestyle import func_read_fe,variable_desc,variable_grouping,Create_EDA_Plots,func_input_user

Sim.dataset_param() # A method to set values of specific parameters for each dataset
Sim.task_param() # A method for specifying which tasks are going to be skipped



st.markdown("##### Read the dataset and Feature Engineering")
variable_grouping(Sim) # A function to group variabls as ordinal, categorical, nuerical
# Read the dataset and Feature Engineering for all models except NeuralNetwork
preprocessor, df_dropna_fe,variables_to_process,feature_names = func_read_fe(Sim,'MinMax')
# Read the dataset and Feature Engineering for NeuralNetwork model
preprocessor_nn, df_dropna_fe_nn,variables_to_process_nn,feature_names_nn = func_read_fe(Sim,'Standard')
st.write('\t\t\t\t\t\t\t\t\t------------------------------')


st.markdown("##### Some Exploratory Data Analysis (EDA)")
# By choosing "Sim.Create_EDAplot" False/True it creates the EDA plots or show the saved plots
if(Sim.Create_EDAplot): 
    Create_EDA_Plots()
else:
    EDA_filename = f'saved/Figs/EDA_{dataset}.png'
    st.image(EDA_filename, caption=" ", use_column_width=True)
st.write('\t\t\t\t\t\t\t\t\t------------------------------')


st.markdown("##### Learning Methods:")
Create_Text_Learning() # Write texts related to learning

# Initialize ML models from saved files
Sim.saved_models = Initialize_Models(Sim)
# Some ML models need to be learned again
if Sim.DoTrain:
    Sim.models2train = Learning_and_Metrics(Sim,df_dropna_fe,df_dropna_fe_nn)
st.write('\t\t\t\t\t\t\t\t\t------------------------------')

for ml in list(Sim.saved_models):
    # Create a dictionary of models, the result plots and scores are going to be shown
    if Sim.DoShowRes[ml]:
        if Sim.DoTrai[ml]: Sim.models2showResult[ml]=Sim.models2train[ml]
        else: Sim.models2showResult[ml]=Sim.saved_models[ml]

    # Create a dictionary of models to be used for prediction
    if Sim.DoPredInpu[ml]:
        if Sim.DoTrai[ml]: Sim.models2predict[ml]=Sim.models2train[ml]
        else: Sim.models2predict[ml]=Sim.saved_models[ml]
st.write('\t\t\t\t\t\t\t\t\t------------------------------')


st.markdown("##### Results:")
for out in Sim.dependent_variables[:1]:
    Plots_Output(out,Sim)
st.write('\t\t\t\t\t\t\t\t\t------------------------------')


st.markdown("### Now Predicting User Condition:")
if Sim.DoPredInput:
    df_input_fe,df_input_fe_nn = func_input_user(preprocessor,preprocessor_nn,variables_to_process,feature_names)
    predict_userInput(Sim,df_input_fe,df_input_fe_nn)
st.write('\t\t\t\t\t\t\t\t\t------------------------------')

Create_Text_End()



