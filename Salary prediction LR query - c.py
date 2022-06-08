# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 17:39:53 2022

@author: QR46
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sal=pd.read_csv(r'C:\Users\qr46\OneDrive - Tesco\Desktop\Python Learning\Linear Regression\Salary_data.csv')

print(sal.head())
print(sal.info())
print(sal.describe())
print(sal.columns)
plt.figure(figsize=(12,6))
#sns.pairplot(sal,x_vars=['YearsExperience'],y_vars=['Salary'],kind='scatter')
plt.xlabel('Years')
plt.ylabel('Salary')
plt.title('Salary Prediction')
X=sal['YearsExperience']
print(X.head())
y=sal['Salary']
print(y.head())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=100)


X_train=X_train[:,np.newaxis]
X_test=X_test[:,np.newaxis]

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

'''
c=[i for i in range (1,len(y_test)+1,1)]
plt.plot(c,y_test,color='r',linestyle='-')
plt.plot(c,y_pred,color='b',linestyle='-')
plt.xlabel('Index')
plt.ylabel('Salary')
plt.title('Prediction')
'''
'''
c=[i for i in range(1,len(y_test)+1,1)]
plt.plot(c,y_test-y_pred,color='green',linestyle='-')
plt.xlabel('index')
plt.ylabel('Error')
plt.title('Error Value')
'''
from sklearn.metrics import r2_score,mean_squared_error
mse=mean_squared_error(y_test,y_pred)

rsq=r2_score(y_test, y_pred)
print('mean squared error:', mse)
print('r square:',rsq)

plt.figure(figsize=(12,6))
plt.scatter(y_test,y_pred,color='r',linestyle='-')

print('Intercept of the model:', lr.intercept_)
print('Coefficient of the line:', lr.coef_)

