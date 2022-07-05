#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 23:24:21 2022

@author: sehtab
"""

import streamlit as st
import pandas as pd


header = st.container()
dataset  = st.container()
features = st.container()
model_training = st.container()

@st.cache 
def get_data(filename):
    df = pd.read_csv(filename)
    return df

with header:
    st.title('Welcome to Student Evaluation System')
    st.text("This software will evaluate student's admission chance using Machine Learning")
    

with dataset:
    st.header('Student-Algorizin Dataset')
    st.text('Found this dataset from Algorizin')
    
    df = get_data('student-algorizin.csv')
    st.write(df.head())
    
    st.subheader("Student's Acceptance Score")
    
    j_dist = pd.DataFrame(df['J'].value_counts())
    st.bar_chart(j_dist)
    
with features:
    st.header('The feature I created')
    
    st.markdown('* **First Feature**')
    st.markdown('* **Second Feature**')
    
    
with model_training:
    st.header('Time to train the model!')
    st.text('Here to choose hyperparemeters of the model & see how the performance is changing')
    
    sel_col, disp_col = st.columns(2)
    max_depth = sel_col.slider('max_depth: ',min_value=5,max_value=30, value=15, step= 5 )
    n_estimators = sel_col.selectbox('How many tress? ',options=[100,200,300,400,500], index=0 )
    reg_alpha = sel_col.slider('reg_alpha ',min_value=1.0,max_value=2.0, value=1.5, step= 0.5 )
    subsample = sel_col.slider('subsample: ',min_value=0.2,max_value=1.0, value=0.8, step= 0.1 )
    input_feature = sel_col.text_input('which feature should be used as the input feature?','A1')
    
    
    sel_col.text('List of Features:')
    sel_col.write(df.columns)
        
    from xgboost import XGBRegressor
    # split the data
    x = df[[input_feature]].values
    y = df[['J']].values
    
    xgb_r = XGBRegressor(max_depth=max_depth,n_estimator=n_estimators,reg_alpha=reg_alpha, subsample=subsample, learning_rate=0.1, objective='reg:linear', booster='gbtree')
    xgb_r.fit(x, y) 
    prediction = xgb_r.predict(y)
    
    from sklearn.metrics import r2_score
    from sklearn import metrics
    import numpy as np
    
    disp_col.subheader('R^2 Score:')
    disp_col.write(r2_score(y, prediction))
    
    disp_col.subheader('Mean Absolute Error:')
    disp_col.write(metrics.mean_absolute_error(y, prediction))
    
    disp_col.subheader('Mean Square Error:')
    disp_col.write(metrics.mean_squared_error(y, prediction))
    
    disp_col.subheader('Root Mean Square Error:')
    disp_col.write(np.sqrt(metrics.mean_absolute_error(y, prediction)))
    
    