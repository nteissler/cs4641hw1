from neuralnet import NeuralNetMLP
from mnist_data import load_mnist
import numpy as np
import matplotlib.pyplot as plt

from wine import X_train, X_test, y_train, y_test

nn = NeuralNetMLP(n_output=3, 
                  n_features=X_train.shape[1], 
                  n_hidden=50, 
                  l2=0.0, 
                  l1=0.0, 
                  epochs=40, 
                  eta=0.001,
                  alpha=0.001,
                  decrease_const=0.00001,
                  minibatches=20, 
                  shuffle=True,
                  random_state=1)

nn.fit(X_train, y_train, print_progress=True)

import sys
# Predict on training Data
y_train_pred = nn.predict(X_train)

if sys.version_info < (3, 0):
    acc = (np.sum(y_train == y_train_pred, axis=0)).astype('float') / X_train.shape[0]
else:
    acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]

print('Training accuracy: %.2f%%' % (acc * 100))

# Predict on Test Data
y_test_pred = nn.predict(X_test)

if sys.version_info < (3, 0):
    acc = (np.sum(y_test == y_test_pred, axis=0)).astype('float') / X_test.shape[0]
else:
    acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]

print('Test accuracy: %.2f%%' % (acc * 100))

batches = np.array_split(range(len(nn.cost_)), 40)
cost_ary = np.array(nn.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]

plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
plt.ylim([0,50])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.tight_layout()
plt.show()

