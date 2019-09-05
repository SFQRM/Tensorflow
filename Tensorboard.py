'''
Tensorboard是Tensorflow自带的可视化曲线分析工具
  运行Tensorboard.py后，会在当前路径下多出一个文件
  其次，在控制台中cd至当前路径
  然后，键入tensorboard --logdir=当前路径
  最后，终端会返回一个网址，并将该网址复制到浏览器中，可视化曲线分析工具会自动刷新
'''

import tensorflow as tf
import numpy as np


# 构造神经层-start #
# 函数参数为：输入数据，输入数据数量，输出数据数量，激励函数类型
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weight'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),
                                  name='W')
            # tf.histogram_summary(layer_name+'/weights', Weights)
            tf.summary.histogram(layer_name+'/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,         # 推荐初始化不为0
                                 name='b')
            # tf.histogram_summary(layer_name + '/biases', biases)
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)      # 构造预测结果
        if activation_function is None:                                 # 如果没有激励函数
            outputs = Wx_plus_b
        else:                                                           # 如果有激励函数
            outputs = activation_function(Wx_plus_b)                    # 则将激励函数作用Wx_plus_b
            # tf.histogram_summary(layer_name + '/outputs', outputs)
            tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs                                                  # 返回该层结果
# 构造神经层-end #


# 创造数据-start #
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise
# 创造数据-end #


with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')


# 创建Tensorflow结构-start #
# 输入层1个神经元，隐藏层10个神经元，输出层1个神经元
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)           # 构造隐藏层
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)         # 构造输出层
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),   # (∑((y_data-prediction)^2))/SizeOfy_data
                                        reduction_indices=[1]))
    # tf.scalar_summary('loss', loss)                               # loss曲线递减说明神经网络学习到了东西
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 定义训练模型

init = tf.initialize_all_variables()                                # 初始化所有变量
# 创建Tensorflow结构-end #


# 创建Tensorflow的会话-start #
sess = tf.Session()                                                 # 创建会话
# merged = tf.merge_all_summaries()                                 # 打包
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("D:\code\TensorFlow\Morvan",sess.graph)
sess.run(init)                                                      # 执行初始化运算
# 创建Tensorflow的会话-end #


# 训练-start #
for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i%50==0:
        result = sess.run(merged, feed_dict={xs:x_data, ys:y_data})
        writer.add_summary(result, i)
# 训练-end #
