import numpy as np
import pandas as pd
import tensorflow as tf

dataset = pd.read_csv('./neural-network/data.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

print(X)