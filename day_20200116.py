import tensorflow as tf
import timeit
import numpy as np
np.set_printoptions(suppress=True)
print(tf.__version__)

#%%
# 此处代码需要使用 tf 2 版本运行
# 1.创建输入张量，并赋初始值
a = tf.constant(2.)
b = tf.constant(4.)
# 2.直接计算， 并打印结果
print('a+b=', a+b)

#%% 观测运行时间
# 创建在 CPU 环境上运算的 2 个矩阵
n = 100
with tf.device('/cpu:0'):
    cpu_a = tf.random.normal([1, n])
    cpu_b = tf.random.normal([n, 1])
    print(cpu_a, cpu_b)

def cpu_run(): # CPU 运算函数
    with tf.device('/cpu:0'):
        c = tf.matmul(cpu_a, cpu_b)
    return c

cpu_time = timeit.timeit(cpu_run, number=500)
print('warmup %.5f 微秒' % (cpu_time))

#%% 自动求导
# y = a*(w^2) + b*w +c
# 创建 4 个张量，并赋值
a = tf.constant(1.)
b = tf.constant(2.)
c = tf.constant(3.)
w = tf.constant(4.)

with tf.GradientTape() as tape:# 构建梯度环境
    tape.watch([w]) # 将 w 加入梯度跟踪列表
    # 构建计算过程，函数表达式
    y = a * w**2 + b * w + c
    # 自动求导
    [dy_dw] = tape.gradient(y, [w])
    print(dy_dw) # 打印出导数

#%% 常用神经网络接口
# TensorFlow 除了提供底层的矩阵相乘、相加等数学函数，还内建了常用神经网络运算函数、
# 常用网络层、 网络训练、 模型保存与加载、 网络部署等一系列深度学习系统的便捷功能。
# 使用 TensorFlow 开发， 可以方便地利用这些功能完成常用业务流程，高效稳定。

#%% 线性回归

# 1 采样数据
data = []# 保存样本集的列表
for i in range(100): # 循环采样 100 个点
    # numpy.random.uniform(low,high,size)
    # 从一个均匀分布[low,high)中随机采样，默认返回一个值
    x = np.random.uniform(-10., 10.) # 随机采样输入 x
    # 采样高斯噪声，高斯分布
    eps = np.random.normal(0., 0.01)
    # 得到模型的输出
    y = 1.477 * x +0.089 + eps
    data.append([x, y])
data = np.array(data)
print(data)

# 2 计算误差
def mse(b, w, points):
    # 根据当前的 w,b 参数计算均方差损失
    totalError = 0
    for i in range(0, len(points)): # 循环迭代所有点
        x = points[i, 0] # 获得 i 号点的输入 x
        y = points[i, 1] # 获得 i 号点的输出 y
        # 计算差的平方，并累加
        totalError += (y - (w * x + b)) ** 2
    # 将累加的误差求平均，得到均方差
    return totalError / float(len(points))

# 3 计算梯度
def step_gradient(b_current, w_current, points, lr):
    # 计算误差函数在所有点上的导数，并更新 w,b
    b_gradient = 0
    w_gradient = 0
    M = float(len(points)) # 总样本数
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # 误差函数对 b 的导数： grad_b = 2(wx+b-y)，参考公式(2.3)
        b_gradient += (2/M) * ((w_current * x + b_current) - y)
        # 误差函数对 w 的导数： grad_w = 2(wx+b-y)*x，参考公式(2.2)
        w_gradient += (2/M) * x * ((w_current * x + b_current) - y)
        # 根据梯度下降算法更新 w',b',其中 lr 为学习率
        new_b = b_current - (lr * b_gradient)
        new_w = w_current - (lr * w_gradient)
    return [new_b, new_w]

# 4 更新梯度
def gradient_descent(points, starting_b, starting_w, lr, num_iterations):
    # 循环更新 w,b 多次
    b = starting_b # b 的初始值
    w = starting_w # w 的初始值
    # 根据梯度下降算法更新多次
    for step in range(num_iterations):
        # 计算梯度并更新一次
        b, w = step_gradient(b, w, np.array(points), lr)
        loss = mse(b, w, points) # 计算当前的均方差，用于监控训练进度
        if step%50 == 0: # 打印误差和实时的 w,b 值
            print(f"iteration:{step}, loss:{loss}, w:{w}, b:{b}")
    return [b, w] # 返回最后一次的 w,b

def main():
    # 加载训练集数据，这些数据是通过真实模型添加观测误差采样得到的
    lr = 0.01  # 学习率
    initial_b = 0  # 初始化 b 为 0
    initial_w = 0  # 初始化 w 为 0
    num_iterations = 1000
    # 训练优化 1000 次，返回最优 w*,b*和训练 Loss 的下降过程
    [b, w] = gradient_descent(data, initial_b, initial_w, lr, num_iterations)
    loss = mse(b, w, data)  # 计算最优数值解 w,b 上的均方差
    print(f'Final loss:{loss}, w:{w}, b:{b}')

if __name__ == "__main__":
    main()


