# -*- coding: utf-8 -*-
"""

@author: USER
"""

import pickle
import os 
import numpy as np
import warnings
import streamlit as st
warnings.filterwarnings('ignore')

MODEL_PATH = os.path.join(os.getcwd(), 'model.pkl')

with open(MODEL_PATH, 'rb') as file:
  model = pickle.load(file)
  

#%%

def item(age,cp,trtbps,chol,thalachh,exang,oldpeak,caa,thall ):
    
    if cp=="Typical angina":
        cp=0
    elif cp=="Atypical angina":
        cp=1
    elif cp=="Non-anginal pain":
        cp=2
    elif cp=="Asymptomatic":
        cp=3
    
    if exang=="Yes":
        exang=1
    elif exang=="No":
        exang=0

    user_input=np.expand_dims([age,cp,trtbps,chol,thalachh,exang,
                               oldpeak,caa,thall],axis=0)

    prediction = model.predict(user_input)

    return prediction

# heading
st.markdown("<h1 style='text-align: center; color: blue;'>HEARTDETECT</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: white;'>...a heart disease detection system</h4><br>", unsafe_allow_html=True)

st.write('In Malaysia, heart disease has become the number 1 killer which almost 1 in every 4 deaths is caused by heart disease. On average, about 140 persons for every 100,000 people will suffer from a heart disease during their lifetime')
st.write('A heart attack or myocardial infarction (MI) is one type of heart disease is a serious medical condition which the blood supply to the heart is suddenly blocked, usually by a blood clot.') 
st.write("Many Malaysians in their 20's and 30's are now suffer from heart attacks.")
st.write("To prevent this accurrence, early detection is important. Thus, this application helps to effectively detect if someone has diabetes using Machine Learning.")

age=st.number_input("Age:")
cp = st.selectbox('Chest Pain Types:',("Typical angina","Atypical angina","Non-anginal pain","Asymptomatic")) 
trtbps=st.number_input('Resting Blood Pressure (in mm Hg):')
chol=st.number_input('Serum Cholestoral in mg/dl:')
thalachh=st.number_input('Maximum Heart Rate Achieved:')
exang=st.selectbox('Exercise Induced Angina',["Yes","No"])
oldpeak=st.number_input('Oldpeak')
caa=st.selectbox('Number of Major Vessels Colored by Flourosopy',range(0,5,1))
thall=st.selectbox('Thalassemia:  0: No Effect,  1: Fixed Defect,  2: Normal,  3: Reversable Defect',
                   range(0,4,1))

pred=item(age,cp,trtbps,chol,thalachh,exang,oldpeak,caa,thall )

if st.button("Predict"):    
  if pred == 1:
    st.error('Warning! You have high risk of getting a heart disease!')
    
  else:
    st.success('You have lower risk of getting a heart disease!')
    st.balloons()

