import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer


# load data-start #
digits = load_digits()
X = digits.data                             # 0~9的手写体图片
y = digits.target
y = LabelBinarizer().fit_transform(y)       # 0~9的标签
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
# load data-end #


# 定义神经层-start #
# 函数参数为：输入数据，输入数据数量，输出数据数量，激励函数类型
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1, out_size])) + 0.1             # 推荐初始化不为0
    Wx_plus_b = tf.matmul(inputs, Weights) + biases                 # 构造预测结果
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:                                 # 如果没有激励函数
        outputs = Wx_plus_b
    else:                                                           # 如果有激励函数
        outputs = activation_function(Wx_plus_b)                    # 则将激励函数作用Wx_plus_b
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs                                                  # 返回该层结果
# 定义神经层-end #


# 输入数据-start #
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])
# 输入数据-end #


# 增加输出层-start #
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)
# 增加输出层-end #


cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)


sess = tf.Session()
merged = tf.summary.merge_all()                     # 打包

# train_writer = tf.train.SummaryWriter("D:\code\TensorFlow\Morvan", sess.graph)
# test_writer = tf.train.summaryWriter("D:\code\TensorFlow\Morvan", sess.graph)
train_writer = tf.summary.FileWriter("D:\code\TensorFlow\Morvan\logs\_train", sess.graph)
test_writer = tf.summary.FileWriter("D:\code\TensorFlow\Morvan\logs\_test", sess.graph)

sess.run(tf.global_variables_initializer())

for i in range(500):
    sess.run(train_step, feed_dict={xs:X_train, ys:y_train, keep_prob:0.5})
    if i%50 == 0:
        train_result = sess.run(merged, feed_dict={xs:X_train, ys:y_train, keep_prob:1})
        test_result = sess.run(merged, feed_dict={xs:X_test, ys:y_test, keep_prob:1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)
