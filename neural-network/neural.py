import numpy as np
import pandas as pd
import tensorflow as tf

dataset = pd.read_csv('./neural-network/data.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

print(X)

# section below encodes 'gender' which is column 2, from 'female' to 1 and from 'male' to 0 - it is up to LabelEncoder to chose the numbers.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

print('encoded gender')
print(X)

# section below encodes 'geography' column (column number 1, which holds country name info) to numbers, because there are more than 2 countries, we need to use onehotencoder 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print('encoded countries')
print(X)

# split data to train and test data.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(len(X_train)) #8000
print(len(X_test))  #2000
print(len(y_train)) #8000
print(len(y_test))  #2000

# scale feature - it will normalize data, not sure why? we should be learning w/o normalizing data, if data is normalized then it is not real. In real world we have data that are outside standard deviation.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

## Everything above is just preparing the data
## Building ANN starts here, when we already have data prepared.
ann = tf.keras.models.Sequential()
# add first hidden layer, with 6 neurons
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#add second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#add output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))