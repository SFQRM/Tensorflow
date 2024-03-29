import tensorflow as tf

# 保存到文件-start #
W = tf.Variable([[1,2,3],
                [3,4,5]],
                dtype=tf.float32,
                name='weights')
b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess,"my_net/save_net.ckpt")
    print("Save to path:", save_path)
# 保存到文件-end #


# 注：tensorflow只能保存变量，还不能保存框架等，如需框架还需要重新构造
