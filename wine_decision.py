from sklearn.tree import DecisionTreeClassifier
from plotter import plot_decision_regions
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np

from wine_data import X_train,  X_test, y_train, y_test
tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=0)

tree.fit(X_train, y_train)

from sklearn.tree import export_graphviz
export_graphviz(tree, out_file = 'wine.dot', feature_names=['Alcohol', 'Color Intensity'])

y_pred_train = tree.predict(X_train)
y_pred_test = tree.predict(X_test)
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file = 'wine.dot', feature_names=['Alcohol', 'Color Intensity'])

y_pred_train = tree.predict(X_train)
y_pred_test = tree.predict(X_test)
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print('Decision Tree train/test accuracy: %.3f/%.3f' % (train_accuracy, test_accuracy))

# set up data for plotting
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(X_train.shape[0],X_combined.shape[0]))
plt.xlabel('Alcohol')
plt.ylabel('Color Intensity')
plt.legend(loc='upper left')
plt.show()

