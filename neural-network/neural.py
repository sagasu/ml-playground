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
#add output layer - only one layer, because the result of the network is 0 or 1 - which stands for if a bank user left bank or is still with the bank
# if there were 3 values in a response - 0,1,2 we would need 3 neurons in output layer, not 2 :) to indicate if each of them has a property that may not be related with each other.
# sigmoid is best of output, because it also gives probability and not only the output
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# set properties for training nn
# binary-crossentropy is good only for two outputs, adam is one of algorithms for setting weights during back propagation
# notice that metrics, can be an array of properties, here we judge only by accuracy
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# train nn
# typically people will use batch_size set to 32
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)