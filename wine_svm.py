from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from plotter import plot_decision_regions

import matplotlib.pyplot as plt
import numpy as np

from wine_data import X_train, X_test, y_train, y_test

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

svm = SVC(kernel='rbf', C=2, random_state=0, gamma=0.5)
svm.fit(X_train_std, y_train)

y_pred = svm.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

y_pred_train = svm.predict(X_train_std)
y_pred_test = svm.predict(X_test_std)
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print('SVM Kernel train/test accuracy: %.3f/%.3f' % (train_accuracy, test_accuracy))

# setup data for plotting
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(X_train_std.shape[0],X_combined_std.shape[0]))
plt.xlabel('Alcohol [standardized]')
plt.ylabel('Color Intensity [standardized]')
plt.legend(loc='upper left')
plt.show()
