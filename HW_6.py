# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 23:50:52 2018

@author: hecha
"""


from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.utils import resample
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()
X = iris.data
y = iris.target


## TASK 1

parameters_array=(1,2,3,4,5,6,7,8,9,10)

out_sample_score = []
in_sample_score = []

tree = DecisionTreeClassifier()

for para in parameters_array:

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=para)

    tree.fit(X_train, y_train)
    y_pred_train = tree.predict(X_train)
    y_pred_test = tree.predict(X_test)
    in_sample_score.append(accuracy_score(y_train, y_pred_train))  
    out_sample_score.append(accuracy_score(y_test, y_pred_test))

print("In sample accuracy(random from 1 to 10):")
print(in_sample_score)
print(" ")
print("Out sample accuracy(random from 1 to 10):")
print(out_sample_score)
print(" ")
print("mean of in sample accuracy: {:.6f}".format(np.mean(in_sample_score)))
print("mean of out sample accuracy: {:.6f}".format(np.mean(out_sample_score)))
print("variance of in sample accuracy: {:.6f}".format(np.var(in_sample_score)))
print("variance of out sample accuracy: {:.6f}".format(np.var(out_sample_score)))

print(" ")
print(" ")


## TASK 2

X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=10)

cv_scores = cross_val_score(tree, X_train, y_train, cv = 10)
print("cv_score(fold from 1 to 10):")
print(cv_scores)
print("mean of cv score: {:.6f}".format(np.mean(cv_scores)))
print("variance of cv score: {:.6f}".format(np.var(cv_scores)))

y_pred = tree.predict(X_test)
out_sample_accuracy = accuracy_score(y_test, y_pred) 
print(" ")
print("out sample accuracy: {:.6f}".format(out_sample_accuracy))


print(" ")


print("**********************************************************************")
print("My name is Chaozhen He")
print("My NetID is: che19")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
print("**********************************************************************")
print("  ")






