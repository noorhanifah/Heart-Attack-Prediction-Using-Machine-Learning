# -*- coding: utf-8 -*-
"""

@author: USER
Age : Age of the patient

Sex : Sex of the patient

cp : Chest Pain type

Value 0: typical angina

Value 1: atypical angina

Value 2: non-anginal pain

Value 3: asymptomatic

trtbps : resting blood pressure (in mm Hg)

chol: cholesterol in mg/dl fetched via BMI sensor

fbs: (fasting blood sugar > 120 mg/dl)

1 = true

0 = false

rest_ecg: resting electrocardiographic results
Value 0: normal

Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)

Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

thalach: maximum heart rate achieved

exang: exercise induced angina

1 = yes

0 = no

old peak: ST depression induced by exercise relative to rest

slp: the slope of the peak exercise ST segment

0 = unsloping

1 = flat

2 = downsloping

caa: number of major vessels (0-3)

thall : thalassemia

0 = null

1 = fixed defect

2 = normal

3 = reversable defect

output: diagnosis of heart disease (angiographic disease status)
0: < 50% diameter narrowing. less chance of heart disease

1: > 50% diameter narrowing. more chance of heart disease
"""

import os 
import pickle 
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
#%%
# 1) Data Loading

CSV_PATH = os.path.join(os.getcwd(),'heart.csv')

df = pd.read_csv(CSV_PATH)

#%%
# 2) Data Inspection/Visualization

df.describe().T
df.dtypes
# All data types are int64 except oldpeak which in float64

# Seperate the categorical and continous data
cat_cols = ['sex','cp','fbs','restecg','exng','slp','caa','thall','output']
con_cols = df.drop(labels=cat_cols,axis=1).columns

for i in con_cols:
    plt.figure
    sns.distplot(df[i])
    plt.show()

df.boxplot()

fig, axes = plt.subplots(5,figsize=(8,20))
for i,c in enumerate(df[con_cols]):
    df[[c]].boxplot(ax=axes[i], vert=False)
    
# Based on the visualized continous data, there are outliers in trtbps(resting blood pressure), 
# and chol(cholesterol), oldpeak columns
    
for i in cat_cols:
    plt.figure()
    sns.countplot(df[i])
    plt.show()
    
df['thall'].value_counts()[0]

# Based on the countplot, sex = 1 has more than twice the people with sex = 0
# Although this dataset shows that there is no null value, but 0 in thall means null and there are 2 null values in thall

corrmat = df.corr()
plt.subplots(figsize=(14,8))
sns.heatmap(corrmat, yticklabels=True, square= False, annot=True)

# Based on the heatmap, there is some correlation between the output and cp,thalachh,exng,oldpeak,slp and caa
# Age and cholesterol usually plays the of contributing to the occurrence of heart attack, however
# the heatmap does not show an high correlation for both of the factor

ax = sns.countplot(x="output", hue="sex", data=df)
ax = sns.countplot(x="output", hue="cp", data=df)
ax = sns.countplot(x="output", hue="exng", data=df)
ax = sns.countplot(x="output", hue="slp", data=df)
ax = sns.countplot(x="output", hue="caa", data=df)
ax = sns.countplot(x="output", hue="fbs", data=df)

# Those who have exercie induced angina has the possibility to get heart attack
# Those who had a downsloping(2) slp has the possibility to get heart attack
# Those with 0 major vessels, caa = 0 have high possibility to get heart attack.

#%%
# 3)Data Cleaning

# View the new data distribution 

df.describe().T

# Handling the null value (0) in thall column
# Use KNN Imputer to fill up NaN values for thall column

df['thall'] = df['thall'].replace(0, np.nan)
df.isna().sum()

col_names= df.columns

knn_imputer = KNNImputer()
df = knn_imputer.fit_transform(df)  # Data is in array form
df = pd.DataFrame(df) #convert data to DataFrame format
df.columns = col_names

df.describe().T

# Change the converted NaN value as a float value is not applicable for this column
df['thall'] = np.floor(df['thall'])

df.isna().sum()

df['trtbps'].max()
df['trtbps'].min()

df['chol'].max()
df['chol'].min()

df['oldpeak'].max()
df['oldpeak'].min()

#%% 
# 4) Features Selection 

# Continous vs categorical data
for i in con_cols:
  lr = LogisticRegression()
  lr.fit(np.expand_dims(df[i],axis=-1),df['output'])
  print(lr.score(np.expand_dims(df[i],axis=-1),df['output']))

# Categorical vs categorical

for i in cat_cols:
    print(i)
    confusion_matrix = pd.crosstab(df[i],df['output']).to_numpy()
    print(cramers_corrected_stat(confusion_matrix))

# The sex,fbs,restecg,slp column will be drop as it has low score

#%%
# 5) Model Training 

X = df.drop(labels=['sex','fbs','restecg','slp','output'],axis=1)
y = df['output']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,
                                               random_state=123)
#LogisticRegression
pipeline_mms_lr = Pipeline([
                            ('Min_Max_Scaler',MinMaxScaler()),
                            ('Logistic_Classifier',LogisticRegression())
                           ]) 

pipeline_ss_lr = Pipeline([
                           ('Standard_Scaler',StandardScaler()),
                           ('Logistic_Classifier',LogisticRegression())
                          ]) 

#Decision Tree
pipeline_mms_dt = Pipeline([
                            ('Min_Max_Scaler',MinMaxScaler()),
                            ('Tree_Classifier',DecisionTreeClassifier())
                           ]) 

pipeline_ss_dt = Pipeline([
                           ('Standard_Scaler',StandardScaler()),
                           ('Tree_Classifier',DecisionTreeClassifier())
                          ]) 

#Random Forest
pipeline_mms_rf = Pipeline([
                            ('Min_Max_Scaler',MinMaxScaler()),
                            ('Random_Forest_Classifier',RandomForestClassifier())
                           ]) 

pipeline_ss_rf = Pipeline([
                           ('Standard_Scaler',StandardScaler()),
                           ('Random_Forest_Classifier',RandomForestClassifier())
                          ]) 

#Gboost
pipeline_mms_gboost = Pipeline([
                             ('Min_Max_Scaler',MinMaxScaler()),
                             ('GBoost_Claasifier',GradientBoostingClassifier())
                            ]) 

pipeline_ss_gboost = Pipeline([
                            ('Standard_Scaler',StandardScaler()),
                            ('GBoost_Claasifier',GradientBoostingClassifier())
                           ]) 

#KNeighbors
pipeline_mms_knn = Pipeline([
                             ('Min_Max_Scaler',MinMaxScaler()),
                             ('KNN_Claasifier',KNeighborsClassifier())
                            ]) 

pipeline_ss_knn = Pipeline([
                            ('Standard_Scaler',StandardScaler()),
                            ('KNN_Claasifier',KNeighborsClassifier())
                           ]) 


#Create a list to store all pipelines
pipelines = [pipeline_mms_lr,pipeline_ss_lr,pipeline_mms_dt,pipeline_ss_dt,
             pipeline_mms_rf,pipeline_ss_rf,pipeline_mms_gboost,
             pipeline_ss_gboost,pipeline_mms_knn,pipeline_ss_knn]

for pipe in pipelines:
    pipe.fit(X_train, y_train)
    
for i,pipe in enumerate(pipelines):  # i is index
    print(pipe.score(X_test, y_test))

best_accuracy = []

for i, pipe in enumerate(pipelines):
  best_accuracy.append(pipe.score(X_test,y_test))

print(pipelines[np.argmax(best_accuracy)])
print(best_accuracy[np.argmax(best_accuracy)])

best_pipe = pipelines[np.argmax(best_accuracy)]

#%%

# 6) Model evaluation

y_true = y_test
y_pred = best_pipe.predict(X_test)

cr = classification_report(y_true,y_pred)
print(cr)

#%%
# 7) Hyper parameter tuning (Grid SearchCV)
# Standard Scalar and Logistic Regression gives the highest score 

pipeline_ss_lr = Pipeline([
                           ('Standard_Scaler',StandardScaler()),
                           ('Logistic_Classifier',LogisticRegression())
                          ]) 

grid_param = [{'Logistic_Classifier__penalty':['l2','elasticnet','l1','none'],
               'Logistic_Classifier__fit_intercept':[True,False],           
               'Logistic_Classifier__random_state':[None,1],
               'Logistic_Classifier__max_iter':[100,200,500],
               'Logistic_Classifier__solver':['saga','sag','liblinear']}]

gridSearch = GridSearchCV(estimator=pipeline_ss_lr,param_grid=grid_param,cv=5,
                          verbose=1,n_jobs=-1)
grid = gridSearch.fit(X_train,y_train)

#%%
# Best parameter
gridSearch.score(X_test,y_test)
print(grid.best_score_)
print(grid.best_params_)

#%%
# 8) Best model saving
best_model = grid.best_estimator_

MODEL_PATH = os.path.join(os.getcwd(),'model.pkl')
with open(MODEL_PATH,'wb') as file:
  pickle.dump(best_model,file)











