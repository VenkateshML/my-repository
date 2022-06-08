# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 08:32:58 2022

@author: QR46
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(r'C:\Users\qr46\OneDrive - Tesco\Desktop\Python Learning\Classified Data.csv',index_col=0)
print(df.head())
print(df.columns)
print(df.info())
print(df.describe())
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))

scaled_features=scaler.transform(df.drop('TARGET CLASS',axis=1))
print(scaled_features)
df_feat=pd.DataFrame(scaled_features,columns=df.columns[:-1])
print(df_feat.head())

from sklearn.model_selection import train_test_split
X=df_feat
y=df['TARGET CLASS']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

error_rate=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',
         markerfacecolor='red',markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

knn=KNeighborsClassifier(n_neighbors=17) 
knn.fit(X_train,y_train) 
pred=knn.predict(X_test) 

print(confusion_matrix(y_test,pred)) 
print('\n')
print(classification_report(y_test, pred))