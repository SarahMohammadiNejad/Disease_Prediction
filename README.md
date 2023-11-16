# Disease_Prediction; Clinical and Lifestyle

To run this application you need to run:
- cd Streamlit_Version
- streamlit run app.py
If you don't have streamlit:
pip install streamlit


### Introduction
Every year millions of people die from cardiovascular disease(CVDs) and different types of cancer 
around the world. Some of the most common risk factors are non-healthy lifestyle, obesity, high 
blood pressure, high cholesterol, long-term stress, diabetes. One can prevent disease by having a 
healthy lifestyle: by being physically active, choosing healthy diet, avoiding smoking, managing 
stress, and getting regular health screens.


##### Main objectives
- Creating machine learning models and the best hyperparameters to classify people for each disease.
- Finding the features with highest impact on each disease.

##### 1. Clinical Dataset
UCI repository link: https://archive.ics.uci.edu/ml/datasets/Heart+Disease. \n 
303 samples, Target: heart disease, 14 featurers: mostly clinical measurements:
**categorical_variables:**  sex, chest_pain_type, rest_ecg, st_slope, thalassemia, fasting_blood_suger, exercise_induced_angina
**numerical_variables:** age, num_major_vessels, max_heart_rate_achieved, cholesterol, st_depression, resting_blood_pressure
- st_slope: ECG readout indicates quality of blood flow.  \n
- thalassemia: Blood disorder, shows blood flow to the heart   \n
- number of major vessels colored by flouroscopy   \n
- st_depression induced by exercise. A measure of abnormality in ECG.

More interesting for clinical purposes, for people who are already diagnosed to have one serious risk factor.

##### 2. Lifestyle Dataset
Centers for Disease Control and Prevention: https://www.cdc.gov/brfss/annual_data/annual_2022.html, (a survey, every year).  
\n Large sample size (450,000), Target: CVD dieseases as well as different type of cancers, 300 features. 
"**Target disease:** attack, stroke, angina, skin cancer, other cancer
**Categorical_variables:** sex, cholesterol, blood pressure, exercise, difficulty walking, marital status, employment status, diabetes.
**Ordinal variables:** age, education, BMI, income, smoke, mental health, alcohol, checkup.
**Note**: Dataset is imbalanced toward the healthy group. RandomUnderSampler, minority_class_count


