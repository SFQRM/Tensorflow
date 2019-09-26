import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 准备数据集-start #
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 准备数据集-end #


# 定义变量-start #
xs = tf.placeholder(tf.float32, [None, 784])                            # xs代表将输入数据表示成张量
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])


def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)


def bias_variable(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)
# 定义变量-end #


# 定义模型-start #
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def convolution_layer(x, W):
    # stride[1, x_movement, y_movement, 1]
    # stride[0]和stride[3]必须为1
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')        # 构造卷积层


def max_pooling_layer_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])                                # 第一层卷积层的权重, patch:5x5 insize(输入图片的厚度):1 outsize:32
b_conv1 = bias_variable([32])                                           # 第一层卷积层的偏置
hiden_conv1 = tf.nn.relu(convolution_layer(x_image, W_conv1) + b_conv1) # 第一层卷积层, output size 28x28x32
hiden_pool1 = max_pooling_layer_2x2(hiden_conv1)                        # 第一层池化层, output size 14x14x32


W_conv2 = weight_variable([5, 5, 32, 64])                               # 第二层卷积层的权重, patch:5x5 insize:32 outsize:64
b_conv2 = bias_variable([64])                                           # 第二层卷积层的偏置
hiden_conv2 = tf.nn.relu(convolution_layer(hiden_pool1, W_conv2) + b_conv2) # 第二层卷积层, output size 14x14x64
hiden_pool2 = max_pooling_layer_2x2(hiden_conv2)                        # 第二层池化层, output size 7x7x64


W_func1 = weight_variable([7*7*64, 1024])
b_func1 = bias_variable([1024])
hiden_pool2_flat = tf.reshape(hiden_pool2, [-1, 7*7*64])                # [n_samples,7,7,64]->>[n_samples,7x7x64]
hiden_func1 = tf.nn.relu(tf.matmul(hiden_pool2_flat,W_func1)+b_func1)
hiden_func1_drop = tf.nn.dropout(hiden_func1, keep_prob)


W_func2 = weight_variable([1024, 10])
b_func2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(hiden_func1_drop, W_func2)+b_func2)
# 定义模型-end #


# 训练模型-start #
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),    # 计算交叉熵
                                              reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)       # 训练

init = tf.global_variables_initializer()                                # 初始化所有变量句柄
sess = tf.Session()                                                     # 创建tf会话
sess.run(init)                                                          # 初始化所有变量

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)                    # 随机抓取训练集中的100个批处理数据点
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    # sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:                                                     # 每训练50次输出一次训练结果
        print(compute_accuracy(mnist.test.images[:1000],
                               mnist.test.labels[:1000]))
# 训练模型-end #
