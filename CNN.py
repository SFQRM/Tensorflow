import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def compute_accuracy(v_xs, v_ys):
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    '''
        # 从截断的正态分布中输出随机值，生成的值遵循具有指定平均值和标准偏差的正态分布。
        tf.truncated_normal(
            shape,                      # tensor的形状
            mean=0.0,                   # 均值
            stddev=1.0,                 # 标准差
            dtype=tf.float32,           # 输出类型
            seed=None,                  # 随机种子
            name=None                   # 操作的名称(可选)
        )
    '''
    initial = tf.truncated_normal(shape, stddev=0.1)                              # 生成符合正态分布的随机值
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride[1, x_movement, y_movement, 1]
    # stride[0]和stride[3]必须为1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    

sess = tf.Session
