# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:05:18 2022

@author: QR46
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv(r'C:\Users\qr46\OneDrive - Tesco\Desktop\Python Learning\Titanic Train.csv')
print(train.head())
#sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')
sns.set_style('whitegrid')
#sns.countplot(x='Survived',hue='Sex', data=train, palette='RdBu_r')
#sns.countplot(x='Survived',hue='Pclass', data=train)
#sns.displot(train['Age'].dropna(), kde=False, bins=30)
#train['Age'].plot.hist(bins=35)
#train['Fare'].hist(bins=40, figsize=(10,4))
#plt.figure(figsize=(10,7))
#sns.boxplot(x='Pclass', y='Age', data=train)
def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass ==2:
            return 29
        else:
            return 24
    else:
        return Age
train['Age']= train[['Age', 'Pclass']].apply(impute_age,axis=1)
train.drop('Cabin', axis=1, inplace=True)
train.dropna(inplace=True)
#sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')
sex=pd.get_dummies(train['Sex'],drop_first=True)
embark=pd.get_dummies(train['Embarked'],drop_first=True)    
train=pd.concat([train,sex,embark],axis=1)
train.drop(['Sex','Embarked', 'Name', 'Ticket','PassengerId'],axis=1,inplace=True)
print(train.head())
X=train.drop('Survived', axis=1)
y=train['Survived']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression(max_iter=10000)
logmodel.fit(X_train,y_train)
predictions=logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))
