import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split


df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol',
                    'Malic Acid', 'Ash', 'Alcalinity of ash',
                    'Magneisum', 'Total Phenols', 'Flavanoids', 'Nonflavanoid phenols',
                    'Proanthocyanins', 'Color Intensity', 'Hue', 'OD280/OD315 of Diluted Wines', 'Proline']
# remove all wine classes with a label of 1, we are doing a binary classification 
#df_wine = df_wine[df_wine['Class label'] != 1]

# we are only classifying on the alcohol and hue features
y = df_wine['Class label'].values
X = df_wine[['Alcohol', 'Color Intensity']].values

le = LabelEncoder()

# encode classifications as 1 and 0 rather than 2 and 3 (original state)
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=1)

