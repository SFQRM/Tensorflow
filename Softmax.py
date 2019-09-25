import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 准备数据集-start #
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 准备数据集-end #


# 定义变量-start #
x = tf.placeholder("float", [None, 784])                # None表示此张量的第一个维度可以是任何长度的
                                                        # 第一维代表照片的数量，第二维每张照片的像素点（18*18）
W = tf.Variable(tf.zeros([784, 10]))                    # W[i]代表权重
b = tf.Variable(tf.zeros([10]))                         # b[i]代表数字i的偏置量
# 定义变量-end #


# 定义回归模型-start #
y = tf.nn.softmax(tf.matmul(x, W) + b)                  # 定义模型：计算概率
# 定义回归模型-end #


# 训练模型-start #
y_ = tf.placeholder("float", [None, 10])                # 定义实际的标签
cross_entropy = -tf.reduce_sum(y_*tf.log(y))            # 计算交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)    # 训练：使用梯度下降算法最小化交叉熵

init = tf.initialize_all_variables()                    # 初始化所有变量
sess = tf.Session()                                     # 创建tf会话
sess.run(init)                                          # 初始化所有变量

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)    # 随机抓取训练数据集中的100个批处理数据点
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})   # 将数据填充至对应变量中，并开始训练
# 训练模型-end #


# 评估模型-start #
    '''
        tf.argmax是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值.
        由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签.
        比如，tf.argmax(y_,1)代表正确的标签.
    '''
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  # 检测预测值是否与真实标签匹配
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) # 定义准确度
# 评估模型-end #


# 结果-start #
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))    # 输出结果
# 结果-end #
