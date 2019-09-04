import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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
    return outputs                                                  # 返回该层结果
# 构造神经层-end #


# 创建实验数据-start #
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)                     # 增加噪音，使得实验数据不是那么完美
y_data = np.square(x_data) - 0.5 + noise
# 创建实验数据-end #


xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])


# 创建Tensorflow结构-start #
# 输入层1个神经元，隐藏层10个神经元，输出层1个神经元
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)           # 构造隐藏层
prediction = add_layer(l1, 10, 1, activation_function=None)         # 构造输出层
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),       # (∑((y_data-prediction)^2))/SizeOfy_data
                                    reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 定义训练模型
init = tf.initialize_all_variables()                                # 初始化所有变量
# 创建Tensorflow结构-end #


# 创建Tensorflow的会话-start #
sess = tf.Session()                                                 # 创建会话
sess.run(init)                                                      # 执行初始化运算
# 创建Tensorflow的会话-end #


# 可视化输出-start #
fig = plt.figure()                                                  # 定义画布
ax = fig.add_subplot(1, 1, 1)                                       # 创建子图
ax.scatter(x_data, y_data)                                          # 绘制真实数据
plt.ion()                                                           # 连续画图
plt.show()                                                          # 展示拟合结果
# 可视化输出-end #


# 训练-start #
for i in range(1000):                                               # 训练1000次
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})        # 开始训练
    if i % 50:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})     # 预测值
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)               # 根据预测值画曲线
        plt.pause(0.1)
# 训练-end #
