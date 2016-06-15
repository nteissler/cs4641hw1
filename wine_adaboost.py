from sklearn.metrics import accuracy_score
from plotter import plot_decision_regions
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

import matplotlib.pyplot as plt
import numpy as np

# import our training and test data
from wine_data import X_train, X_test, y_train, y_test


tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0)
ada = AdaBoostClassifier(base_estimator=tree, n_estimators=40, learning_rate=0.01, random_state=0)
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

# setup either ada plot or tree plot
# change the classifier parameter of plot_decision_regions to change plot
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, classifier=ada, test_idx=range(X_train.shape[0],X_combined.shape[0]))
plt.xlabel('Alcohol')
plt.ylabel('Color Intensity')
plt.legend(loc='upper left')
plt.show()
