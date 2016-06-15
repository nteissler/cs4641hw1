from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from plotter import plot_decision_regions

import matplotlib.pyplot as plt
import numpy as np

from mnist_data import load_mnist

X_train, y_train = load_mnist('mnist', kind='train', abbrv=True)
X_test, y_test = load_mnist('mnist', kind='t10k', abbrv=True)

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

svm = SVC(kernel='poly', C=2, random_state=0, gamma=0.5)
svm.fit(X_train_std, y_train)

y_pred = svm.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

y_pred_train = svm.predict(X_train_std)
y_pred_test = svm.predict(X_test_std)
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print('SVM Kernel train/test accuracy: %.3f/%.3f' % (train_accuracy, test_accuracy))
