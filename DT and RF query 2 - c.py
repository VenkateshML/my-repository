# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 18:20:05 2022

@author: QR46
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

loans=pd.read_csv(r'C:\Users\qr46\OneDrive - Tesco\Desktop\Python Learning\loan_data.csv')
print(loans.head())
print(loans.info())
print(loans.describe())

'''loans[loans['credit.policy']==1]['fico'].hist(bins=35,color='blue',label='Credit Policy =1',alpha=0.6)
loans[loans['credit.policy']==0]['fico'].hist(bins=35,color='red',label='Credit Policy = 0',alpha=0.6)
plt.legend()
plt.xlabel("FICO")
'''

'''
loans[loans['not.fully.paid']==1]['fico'].hist(bins=35,color='blue',label='not fully paid =1',alpha=0.6)
loans[loans['not.fully.paid']==0]['fico'].hist(bins=35,color='red',label='not fully paid= 0',alpha=0.6)
plt.legend()
plt.xlabel("FICO")
'''
'''
plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')
'''

#sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')
'''
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Set1')
'''
cat_feats=['purpose']
final_data=pd.get_dummies(loans,columns=cat_feats,drop_first=True)
print(final_data.info())
print(final_data.head())

from sklearn.model_selection import train_test_split
X=final_data.drop('not.fully.paid',axis=1)
y=final_data['not.fully.paid']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)

prediction=dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,prediction))
print(confusion_matrix(y_test, prediction))


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=300)
rfc.fit(X_train,y_train)

rfc_pred=rfc.predict(X_test)
print(classification_report(y_test,rfc_pred))
print(confusion_matrix(y_test,rfc_pred))
