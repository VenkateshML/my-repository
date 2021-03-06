# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 20:13:48 2022

@author: QR46
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
knn_data=pd.read_csv(r'C:\Users\qr46\OneDrive - Tesco\Desktop\Python Learning\KNN_Project_Data.csv')

print(knn_data.head())

#sns.pairplot(knn_data,hue='TARGET CLASS')

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(knn_data.drop('TARGET CLASS',axis=1))

scaled_features=scaler.transform(knn_data.drop('TARGET CLASS',axis=1))
print(scaled_features)
knn_data_feat=pd.DataFrame(scaled_features,columns=knn_data.columns[:-1])
print(knn_data_feat.head())

from sklearn.model_selection import train_test_split
X=knn_data_feat
y=knn_data['TARGET CLASS']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test, pred))

error_rate=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',
         markerfacecolor='red',markersize=10)
plt.title('Error rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

knn=KNeighborsClassifier(n_neighbors=30) 
knn.fit(X_train,y_train) 
pred=knn.predict(X_test) 

print(confusion_matrix(y_test,pred)) 
print('\n')
print(classification_report(y_test, pred))