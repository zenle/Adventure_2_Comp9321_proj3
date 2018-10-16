import tensorflow as tf
import pandas as pd
import numpy as np
#import sklearn
from sklearn import metrics

dfsub = pd.read_csv('dfsub.csv', index_col=0)
print(dfsub.shape)
train_X = dfsub.drop(['price'], axis = 1).values
print(train_X.shape)
train_Y = dfsub['price'].tolist()

n_samples = train_X.shape[0]
X = tf.placeholder('float', [None,8])
Y = tf.placeholder(dtype=tf.float32)

