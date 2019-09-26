import tensorflow as tf
import numpy as np

# 恢复数据-start #
# 与之前保存数据的shape和type一定要一致
W = tf.Variable(np.arange(6).reshape((2,3)),
                dtype=tf.float32,
                name="weights")
b = tf.Variable(np.arange(3).reshape((1,3)),
                dtype=tf.float32,
                name="biases")

# 无需定义init!!

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "my_net/save_net.ckpt")
    print("weight:", sess.run(W))
    print("biases:", sess.run(b))
# 恢复数据-end #


# OUTPUT:
# weight: [[1. 2. 3.]
#  [3. 4. 5.]]
# biases: [[1. 2. 3.]]


# 注：tensorflow只能保存变量，还不能保存框架等，如需框架还需要重新构造
