import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#创建连接session，利用高效的C++后端来进行计算。
sess = tf.InteractiveSession()

#权重初始化时加入少量的噪声，stddev标准差
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#用一个较小的正数来初始化偏置项
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#定义卷积，x输入，W过滤，strides步长，padding边距位0
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#定义池化，取2x2中的max
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
#占位符
y_ = tf.placeholder("float", shape=[None, 10])
x = tf.placeholder("float", shape=[None, 784])

#对图片reshape
x_image = tf.reshape(x, [-1,28,28,1])

#卷积池化一
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#卷积池化二
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#密集连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#为了减少过拟合，输出层之前加入dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


#评估模型
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
#训练方式
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#评估准确率，相同为1，不同为0
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#[1,0,1,1]为0.75
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#初始化所有变量
sess.run(tf.initialize_all_variables())

for i in range(20000):
  #一批取50图片，返回一个元组
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print ("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print ("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))