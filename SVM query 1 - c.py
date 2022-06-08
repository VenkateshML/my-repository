# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 13:04:34 2022

@author: QR46
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
print(cancer.keys())
print(cancer['DESCR'])
df_feat=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
print(df_feat.head(2))
print(df_feat.info())
print(cancer['target'])
print(cancer['target_names'])

from sklearn.model_selection import train_test_split
X=df_feat
y=cancer['target']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=101)

from sklearn.svm import SVC
model=SVC()
model.fit(X_train,y_train)
print(SVC)

predictions=model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid=GridSearchCV(SVC(),param_grid,verbose=3)
grid.fit(X_train,y_train)
print(grid.best_params_)
print(grid.best_estimator_)
grid_predictions=grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))

