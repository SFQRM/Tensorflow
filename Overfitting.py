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


# 构造神经层-start #
# 函数参数为：输入数据，输入数据数量，输出数据数量，激励函数类型
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1, out_size])) + 0.1             # 推荐初始化不为0
    Wx_plus_b = tf.matmul(inputs, Weights) + biases                 # 构造预测结果
    if activation_function is None:                                 # 如果没有激励函数
        outputs = Wx_plus_b
    else:                                                           # 如果有激励函数
        outputs = activation_function(Wx_plus_b)                    # 则将激励函数作用Wx_plus_b
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs                                                  # 返回该层结果
# 构造神经层-end #



