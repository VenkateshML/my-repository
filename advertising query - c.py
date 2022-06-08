# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 12:55:13 2022

@author: QR46
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ad_data=pd.read_csv(r'C:\Users\qr46\OneDrive - Tesco\Desktop\Python Learning\advertising.csv')

print(ad_data.head())
#sns.heatmap(ad_data.isnull(),yticklabels=False, cbar=False, cmap='viridis')
print(ad_data.columns)
print(ad_data.info())
print(ad_data.describe())
#ad_data['Age'].plot.hist(bins=35)
#sns.jointplot(data=ad_data,x='Age',y='Area Income')
#sns.jointplot(data=ad_data, x='Age', y='Daily Time Spent on Site', kind='kde',color='red')
#sns.jointplot(data=ad_data, x='Daily Time Spent on Site', y='Daily Internet Usage', color='green')
#sns.pairplot(ad_data, hue='Clicked on Ad')

X=ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage','Male']]
y=ad_data['Clicked on Ad']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train, y_train)

predictions=logmodel.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))