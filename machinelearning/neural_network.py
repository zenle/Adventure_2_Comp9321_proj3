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

h = 20
W = tf.Variable(tf.random_normal([8, h], 0, 0.1))
b = tf.Variable(tf.zeros([h]) + 0.1)
W2 = tf.Variable(tf.random_normal([h, 1], 0, 0.1))
b2 = tf.Variable(tf.zeros([1])+0.1)
L1 = tf.matmul(X, W) + b
pred = tf.matmul(L1, W2) + b2
loss = tf.reduce_mean(tf.square(pred-Y))
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    # 初始化所有变量
    max_steps = 10
    show_step = 5
    initialize = tf.global_variables_initializer().run()
    #迭代训练
    for step in range(max_steps):
        for x,y in zip(train_X,train_Y):
            #print(x, y)
            sess.run(train_step,feed_dict={X:[x],Y:[y]})
        if step % show_step == 0:
            #计算模型在数据集上的损失
            print('*'*40)
            #print(pred)
            train_Xcp = train_X
            ycp = train_Y
            #np.shuffle(train_Xcp)
            #np.shuffle(ycp)
            step_loss = sess.run([loss],feed_dict={X:train_Xcp[:200],Y:ycp[:200]} )
            print("step:",step,"-step loss:%.4f",step_loss[0])
    #计算最终的Loss
    #train`_loss = sess.run([loss],feed_dict={X:train_X,Y:train_Y})
    #print("train loss:%.4f",train_loss)
    #X_test = tf.placeholder('float', [None,8])
    X_test = pd.read_csv('df_test.csv', index_col=0)
    Y_test = pd.read_csv('ytest.csv', index_col=0)
    #Y_pred = tf.placeholder(dtype=tf.float32)
    Y_pred = sess.run([pred], feed_dict={X:X_test})

    print('%'*40)
    print(metrics.mean_squared_error(Y_test, Y_pred))
    #sklearn.metrics
    #输出参数
    #print("weights:",sess.run(W),"-bias:",sess.run(b))
    #plt.plot(train_X,train_Y,"ro",label="original data")
    #plt.plot(train_X,sess.run(pred,feed_dict={X:train_X}),label="predict data")
    #plt.legend(loc="upper left")
    #plt.show()