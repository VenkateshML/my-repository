# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 17:52:23 2022

@author: QR46
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv(r'C:\Users\qr46\OneDrive - Tesco\Desktop\Python Learning\kyphosis.csv')
print(df.head())
#sns.pairplot(df,hue='Kyphosis')

from sklearn.model_selection import train_test_split
X=df.drop('Kyphosis',axis=1)
y=df['Kyphosis']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions=dtree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)

rfc_pred=rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))