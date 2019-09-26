import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 准备数据集-start #
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)       # 准备数据
# 准备数据集-end #


# 定义变量与函数-start #
xs = tf.placeholder(tf.float32, [None, 784])        # 28*28
ys = tf.placeholder(tf.float32, [None, 10])                         # 标签


# 函数参数为：输入数据，输入数据数量，输出数据数量，激励函数类型
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1, out_size])) + 0.1             # 推荐初始化不为0
    Wx_plus_b = tf.matmul(inputs, Weights) + biases                 # 构造预测结果
    if activation_function is None:                                 # 如果没有激励函数
        outputs = Wx_plus_b
    else:                                                           # 如果有激励函数
        outputs = activation_function(Wx_plus_b)                    # 则将激励函数作用Wx_plus_b
    return outputs                                                  # 返回该层结果


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})              # y的预测值
    correct_prediction = tf.equal(tf.argmax(y_pre, 1),              # y的预测值
                                  tf.argmax(v_ys, 1))               # y的实际值 作比较 得到布尔结果
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 布尔结果数值化后求均值，得到精确度
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})     # 调用accuracy计算预测结果
    return result
# 定义变量与函数-end #


# 构造模型-start #
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)       # 构造输出层
# 构造模型-end #


# 训练模型-start #
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))     # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()                                # 初始化所有变量
sess = tf.Session()                                                     # 创建Tensorflow的会话
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)                    # 每次选取100张图片训练
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
# 训练模型-end #

