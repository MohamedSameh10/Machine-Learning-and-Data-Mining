import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
""""
dataset = pd.read_csv("cardio_train.csv",sep=";")
#print(dataset.shape)
X = dataset[dataset.columns[:12]]
y = dataset[dataset.columns[-1]]
#print(x)
#print(x.shape)
#print(y)
"""
dataset = pd.read_csv("ass3.csv")
#print(dataset.shape)
X = dataset[dataset.columns[:5]]
y = dataset[dataset.columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y)

decision_tree = DecisionTreeClassifier()
decision_tree = decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)

#Evaluating
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Visualizing the tree
fig = plt.figure()
_ = tree.plot_tree(decision_tree)
fig.savefig("decistion_tree.png")
#text_representation = tree.export_text(decision_tree)
#print(text_representation)