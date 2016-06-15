from sklearn.tree import DecisionTreeClassifier
from plotter import plot_decision_regions
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np

from mnist_data import load_mnist

X_train, y_train = load_mnist('mnist', kind='train')
X_test, y_test = load_mnist('mnist', kind='t10k')

tree = DecisionTreeClassifier(criterion='gini', max_depth=16, random_state=0)

tree.fit(X_train, y_train)

y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)

print('Deciscion Tree Accuracies train/test: %.3f/%.3f' % (tree_train, tree_test))

from sklearn.tree import export_graphviz
export_graphviz(tree, out_file = 'digit.dot')
