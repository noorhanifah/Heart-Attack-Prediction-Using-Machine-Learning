# -*- coding: utf-8 -*-
"""

@author: USER
"""


import os 
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score


MODEL_PATH = os.path.join(os.getcwd(), 'model.pkl')

with open(MODEL_PATH, 'rb') as file:
  model = pickle.load(file)
  
#%%
CSV_PATH = os.path.join(os.getcwd(),'test.csv')

df = pd.read_csv(CSV_PATH)
df_test = df.drop(labels='output',axis=1)

print(model.predict(df_test))

#%%

y_pred = model.predict(df_test)
y_test = df['output']

print(accuracy_score(y_test,y_pred))