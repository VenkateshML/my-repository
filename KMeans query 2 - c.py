# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 22:26:08 2022

@author: QR46
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r'C:\Users\qr46\OneDrive - Tesco\Desktop\Python Learning\College_Data.csv',
               index_col=0)
print(df.head())
print(df.info())
print(df.describe())

'''
sns.lmplot(x='Room.Board',y='Grad.Rate',data=df,hue='Private',fit_reg=False,
           palette='coolwarm',size=6,aspect=1)

sns.lmplot(x='Outstate',y='F.Undergrad',data=df,hue='Private',fit_reg=False,
           height=6,aspect=1)
'''
#g=sns.FacetGrid(df,hue='Private',palette='coolwarm')
#g=g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)

print(df[df['Grad.Rate']>100])
df['Grad.Rate']['Cazenovia College']=100

print(df[df['Grad.Rate']>100])
#g=sns.FacetGrid(df,hue='Private',palette='coolwarm')
#g=g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=2)
kmeans.fit(df.drop('Private',axis=1))

print(kmeans.cluster_centers_)
def converter(private):
    if private =='Yes':
        return 1
    else:
        return 0
    
df['Cluster']=df['Private'].apply(converter)
print(df.head())

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))