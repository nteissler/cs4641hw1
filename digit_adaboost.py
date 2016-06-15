from sklearn.metrics import accuracy_score
from plotter import plot_decision_regions
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

import matplotlib.pyplot as plt
import numpy as np

# import the digit data 
from mnist_data import load_mnist

X_train, y_train = load_mnist('mnist', kind='train')
X_test, y_test = load_mnist('mnist', kind='t10k')

tree = DecisionTreeClassifier(criterion='entropy', max_depth=16, random_state=0)
ada = AdaBoostClassifier(base_estimator=tree, n_estimators=10, learning_rate=0.01, random_state=0)
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))


ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print('Ada boost train/test accuracies %.3f/%.3f' % (ada_train, ada_test))
