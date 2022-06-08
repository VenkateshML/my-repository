# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 17:08:55 2022

@author: QR46
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

iris=sns.load_dataset('iris')
print(iris.keys())
print(iris.head(2))

#sns.pairplot(iris,hue='species',palette='Dark2')
#setosa=iris[iris['species']=='setosa']
#sns.kdeplot(setosa['sepal_width'], setosa['sepal_length'],cmap='plasma',
#            shade=True,shade_lowest=False)


from sklearn.model_selection import train_test_split
X=iris.drop('species',axis=1)
y=iris['species']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

from sklearn.svm import SVC
svc_model=SVC()
svc_model.fit(X_train,y_train)

predictions=svc_model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid=GridSearchCV(SVC(),param_grid,verbose=2)
grid.fit(X_train,y_train)
print(grid.best_params_)
print(grid.best_estimator_)
grid_predictions=grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))