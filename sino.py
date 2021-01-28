# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
import streamlit as st

bank = pd.read_csv("bank-full.csv", delimiter=(";"))

"""ouliers age"""
outliers_age_up = bank[bank['age'] > bank['age'].mean() + 3 * bank['age'].std()]
outliers_age_down = bank[bank['age'] < bank['age'].mean() - 3 * bank['age'].std()]

"""outliers balance"""
outliers_bal_up = bank[bank['balance'] > bank['balance'].mean() + 3 * bank['balance'].std()]
outliers_bal_down = bank[bank['balance'] < bank['balance'].mean() - 3 * bank['balance'].std()]

"""outliers duration"""
outliers_dur_up = bank[bank['duration'] > bank['duration'].mean() + 3 * bank['duration'].std()]
outliers_dur_down = bank[bank['duration'] < bank['duration'].mean() - 3 * bank['duration'].std()]

"""boxplot balance"""
boxplot_balance = bank.boxplot(column=['balance'])

"""boxplot duration"""
boxplot_duration = bank.boxplot(column=['duration'])

pdays_unique = bank['pdays'].value_counts()
jobs_unique = bank['job'].value_counts()

"""regresão logistica"""
X = bank[['age', 'balance', 'pdays', 'job']]
y = bank['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['act'], colnames=['pred'])
sn.heatmap(confusion_matrix, annot=True)

accuracy = metrics.accuracy_score(y_test, y_pred)

"""redes neuronais"""
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
nn.fit(X, y)
nn.predict(X)
nn.predict_proba(X)
nn_score = round(nn.score(X, y), 2)

"""naive bayes"""
label_enc = preprocessing.LabelEncoder()
y_enc = label_enc.fit_transform(y)

age_y = list(zip(bank["age"], y_enc))

nb = GaussianNB()
nb.fit(age_y, y_enc)
nb_predict = nb.predict([[50,1]])


"""regressão logistica sem dados maus"""
bank2 = bank[bank.age < 70]
bank3 = bank2[bank2.duration < 700]

X2 = bank3[['age', 'balance', 'duration']]
y2 = bank3['y']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.25, random_state=0)

logistic_regression = LogisticRegression()
logistic_regression.fit(X2_train, y2_train)
y2_pred = logistic_regression.predict(X2_test)

confusion_matrix2 = pd.crosstab(y2_test, y2_pred, rownames=['act2'], colnames=['pred2'])
sn.heatmap(confusion_matrix2, annot=True)

accuracy2 = metrics.accuracy_score(y2_test, y2_pred)