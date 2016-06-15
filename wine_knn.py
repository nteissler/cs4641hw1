from sklearn.neighbors import KNeighborsClassifier
from plotter import plot_decision_regions
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

from wine_data import X_train, X_test, y_train, y_test

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

neighbors = 14
knn = KNeighborsClassifier(n_neighbors=neighbors, p=2, metric='minkowski')

knn.fit(X_train_std, y_train)

y_pred_train = knn.predict(X_train_std)
y_pred_test = knn.predict(X_test_std)
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
print('KNN k=%d: train/test accuracy: %.3f/%.3f' % (neighbors, train_accuracy, test_accuracy))

# Setup graph data
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(105,150))

# loop to find value of k
for k in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=k, p=2, metric='minkowski')
    knn.fit(X_train_std, y_train)
    y_pred_train = knn.predict(X_train_std)
    y_pred_test = knn.predict(X_test_std)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print('KNN k=%d: train/test accuracy: %.3f/%.3f' % (k, train_accuracy, test_accuracy))


plt.xlabel('Alcohol [standardized]')
plt.ylabel('Color Intensity [standardized]')
plt.show()
