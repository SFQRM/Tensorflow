import tensorflow as tf
import numpy as np

# 创造数据
x_data = np.random.rand(100).astype(np.float32)
y_data = 0.1*x_data + 0.3

# 创建tensorflow结构-start
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))    # 定义权重
biases = tf.Variable(tf.zeros([1]))                         # 定义偏置

y = Weights*x_data + biases                                 # 构造预测值

loss = tf.reduce_mean(tf.square(y-y_data))                  # 构造损失函数
optimizer = tf.train.GradientDescentOptimizer(0.5)          # 定义优化器
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()                        # 初始化变量
# init = tf.global_variables_initializer()                  # 初始化变量的另一种方式
# 创建tensorflow结构-end

# 创建tensorflow的会话-start
sess = tf.Session()                                         # 创建会话
sess.run(init)                                              # 计算入口
# 创建tensorflow的会话-end

# 训练-start
for step in range(201):
    sess.run(train)
    if step%20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
# 训练-end


"""
结果输出：
    0 [-0.2255939] [0.6834962]
    20 [-0.01578198] [0.36403427]
    40 [0.06744248] [0.31800625]
    60 [0.09084495] [0.3050633]
    80 [0.09742564] [0.3014238]
    100 [0.0992761] [0.30040038]
    120 [0.09979646] [0.30011258]
    140 [0.09994276] [0.30003166]
    160 [0.09998392] [0.30000892]
    180 [0.09999546] [0.30000252]
    200 [0.09999871] [0.30000073]
"""
