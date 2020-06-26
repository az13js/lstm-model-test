# -*- coding: UTF-8 -*-

# 出于通用性的考虑，模型假设可以处理format.py的任何宽度数据，不同数据集宽度不同一定
# 可以通过参数调整。

import tensorflow as tf

# 数据宽度，序列处理长度和模型输出数，出于通用性考虑，这些参数写在这里，可以根据不同的
# 数据集灵活调整
data_width = 14
model_sequence_length = 30
model_final_output_num = 6

# 这里建立一个简单的模型演示 LSTM 层的特性
print('Build model')
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.LSTM(
    # 一次输入的序列长度是多少，第一个元素的值就是多少
    # 表格有多少列，这里第二个元素的值就是多少
    input_shape = (model_sequence_length, data_width),
    # 想要使用多少个LSTM并行处理，这个值就是多少
    units = 5,
    activation = "tanh",
    # recurrent activation是作用于 C 的计算，要模拟原文的话需要设置为sigmoid
    recurrent_activation = "sigmoid",
    # 是否增加一个偏置向量。模拟原文的话设置为 True。
    use_bias = True,
    # 用来初始化内核权重矩阵，用于对输入进行线性转换
    kernel_initializer = "glorot_uniform",
    # 回归计算的内核权重矩阵初始化方法，用于给回归状态进行线性转换 orthogonal 是默认
    # 的值
    recurrent_initializer = "orthogonal",
    # 给偏置进行初始化操作的方法，默认值是 zeros
    bias_initializer = "zeros",
    # 如果设置为 True 会给遗忘门增加一个 bias 参数，同时强制设置bias_initializer为
    # zeros
    unit_forget_bias = True,
    # 内核权重归一化的方法，默认为 None
    kernel_regularizer = None,
    # 回归权重矩阵归一化方法，默认None
    recurrent_regularizer = None,
    # 用于偏置矩阵的归一化方法，默认None
    bias_regularizer = None,
    # 给 LSTM 输出的归一化方法，默认None
    activity_regularizer = None,
    # 用于内核参数矩阵的约束函数，默认None
    kernel_constraint = None,
    # 回归参数矩阵的约束函数，默认None
    recurrent_constraint = None,
    # 偏置参数矩阵的约束函数，默认为None
    bias_constraint = None,
    # 使多少比重的神经元输出（unit的输出）激活失效，默认为0，模仿原文为0
    dropout = 0.5,
    # recurrent_dropout是给递归状态 C 设置的Dropout参数
    recurrent_dropout = 0.0,
    # 实现方式。1会将运算分解为若干小矩阵的乘法加法运算，2则相反。这个参数不用情况下
    # 会使得程序具有不同的性能表现。
    implementation = 2,
    # return_sequences 是否返回全部输出的序列。False否，True返回全部输出，框架默认
    # False，模拟原文可以考虑设置为True
    # 如果只希望LSTM最后一个序列计算输出被下面的层计算，那么设为False
    return_sequences = False,
    # 是否返回LSTM的中间状态，框架默认False，模拟原文可以设置为False。这里返回的状态
    # 是最后计算后的状态
    return_state = False,
    # 是否反向处理。设置为诶True则反向处理输入序列以及返回反向的输出。默认为False
    go_backwards = False,
    # 默认为False，如果设置为True，每一个批的索引i代表的样本的最后一个状态量C，将会作
    # 为初始化状态，初始化下一个索引i的批次样本
    stateful = False,
    # 是否展开。把LSTM展开计算会加速回归计算的过程，但是会占用更多内存，建议在小序列上
    # 展开。默认为False。
    unroll = True
))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(
    units = model_final_output_num,
    activation = "tanh",
    use_bias = True,
))

# 打印调试信息
model.summary()

print('Save "LSTM.model"')
tf.keras.models.save_model(
    model,
    'LSTM.model',
    overwrite = True,
    include_optimizer = False
)
