# -*- coding: utf-8 -*-
import tensorflow as tf
#MNISTの元データを読み込みます
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#入力層の初期化。28×28ピクセルのデータであるため、784の入力層を準備。
x = tf.placeholder("float", [None, 784])
#重みベクトルの初期化。入力層784、出力層10(0~9の数字を見分ける)
W = tf.Variable(tf.zeros([784,10]))
#出力に対する誤差項。
b = tf.Variable(tf.zeros([10]))
#確率にして表現するためソフトマックス関数をかけたものを最終的な出力表現とする。
y = tf.nn.softmax(tf.matmul(x,W) + b)
#正解データを格納する変数を宣言
y_ = tf.placeholder("float", [None,10])
#コスト関数。クロスエントロピー関数を使っています。言ってしまえば学習のときにどれだけ間違えたかを算出する関数です。
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#そのコスト関数を最小化するように値を更新します。学習係数は0.01。誤差逆伝播法。
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#初期化
init = tf.initialize_all_variables()
#セッションというものが計算の流れを管理してくれるらしい
sess = tf.Session()
sess.run(init)
#実際の学習。1000回学習します。
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#出力層の中で一番大きく出力しているところをargmaxで返します。
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
