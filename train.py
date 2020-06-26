# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
import csv
import random
import os
# 如果引入图形弹窗失败，则注释掉，然后保证graph为False来训练
import matplotlib.pyplot as plt

# 出于通用性考虑，任何数据宽度和序列长度变化的改变都可以通过这里的配置得到调整
# 定义模型序列输入有多长
model_sequence_length = 30
# 定义多少比重的数据会被用于训练。出现小数的时候，往下取整。分割数据前会进行随机打乱
train_data_rate = 0.5
# 学习速度
learning_rate = 0.001
# 每轮训练
epochs = 1
# 一共多少轮
epochs_total = 10000
# 是否使用图形。纯命令行时设置为False
graph = True

# 处理输入数据的方法
def input_data_processor(input_data_row):
    return input_data_row

# 处理输出数据的方法
def output_data_processor(output_data_row):
    # 只有前面6个具有预测的价值
    output_data = []
    for i in range(6):
        output_data.append(output_data_row[i])
    return output_data

all_data_from_csv = []
# 从csv文件中读取训练用的数据，顺序
print('Step 1 : Read data from dataset_the_end.csv')
line_num = 1
with open('dataset_the_end.csv', newline='') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        data_row = []
        for item in row:
            data_row.append(float(item))
        all_data_from_csv.append(data_row)
        #all_data_from_csv.append([line_num, line_num])
        #if line_num == 7:
        #    break
        line_num = line_num + 1
print('Step 1 : Read data from dataset_the_end.csv - OK.')

#print('Len=')
#print(len(all_data_from_csv))
#exit()

length_of_all_data_from_csv = len(all_data_from_csv)
if length_of_all_data_from_csv < model_sequence_length + 2:
    print('insufficient')
    print(length_of_all_data_from_csv)
    exit()

print('Step 2 : Build data for model')
# 可以用来给模型作为输入数据使用的数据
all_data_for_model_inputs = []
length_of_all_data_for_model_inputs = length_of_all_data_from_csv - model_sequence_length
for i in range(length_of_all_data_for_model_inputs):
    input_data = []
    output_data = []
    for j in range(model_sequence_length):
        input_data.append(input_data_processor(all_data_from_csv[i + j].copy()))
    output_data.append(output_data_processor(all_data_from_csv[i + model_sequence_length].copy()))
    all_data_for_model_inputs.append({'input': input_data, 'output': output_data})
print('Step 2 : Build data for model - OK')

# 原始输入数据可以不要了
all_data_from_csv = []

print('Step 3 : Shuffle data')
random.shuffle(all_data_for_model_inputs)
print('Step 3 : Shuffle data - OK')

print('Step 4 : Build train dataset and test dataset')
train_dataset_inputs = []
train_dataset_outputs = []
test_dataset_inputs = []
test_dataset_outputs = []
train_dataset_length =  int(length_of_all_data_for_model_inputs * train_data_rate)
test_dataset_length = length_of_all_data_for_model_inputs - train_dataset_length
print("         The length of train-data is %d" % (train_dataset_length))
print("         The length of test-data is %d" % (test_dataset_length))
if train_dataset_length <= 0 or test_dataset_length <= 0:
    print('train_dataset_length <= 0 or test_dataset_length <= 0')
    exit()
for i in range(train_dataset_length):
    train_dataset_inputs.append(all_data_for_model_inputs[i]['input'])
    train_dataset_outputs.append(all_data_for_model_inputs[i]['output'][0])
for i in range(test_dataset_length):
    test_dataset_inputs.append(all_data_for_model_inputs[train_dataset_length + i]['input'])
    test_dataset_outputs.append(all_data_for_model_inputs[train_dataset_length + i]['output'][0])
print('Step 4 : Build train dataset and test dataset - OK')

# 可以不需要了
all_data_for_model_inputs = []

print('Step 5 : Load and compile model')
model = tf.keras.models.load_model('LSTM.model')
model.compile(
    optimizer = tf.keras.optimizers.Adam(lr = learning_rate, beta_1 = 0.9, beta_2 = 0.999),
    loss = "mean_squared_error"
    # metrics = ['accuracy']
)
model.summary()
print('Step 5 : Load and compile model - OK')

print('Step 6 : Train model')
train_np_array_x = np.array(train_dataset_inputs, dtype=np.float64)
train_np_array_y = np.array(train_dataset_outputs, dtype=np.float64)
test_np_array_x = np.array(test_dataset_inputs, dtype=np.float64)
test_np_array_y = np.array(test_dataset_outputs, dtype=np.float64)
train_dataset_inputs = []
train_dataset_outputs = []
test_dataset_inputs = []
test_dataset_outputs = []

# 保存模型和训练结果的文件夹
if False == os.path.exists('models'):
    os.makedirs('models')

train_history_x = []
train_history_loss = []
train_history_val_loss = []

with open('models' + os.sep + 'models_loss.csv', 'a+', encoding='UTF8', newline='') as f:
    csv_write = csv.writer(f)
    csv_write.writerow(['id', 'loss', 'val_loss', 'file'])

# 进入交互式绘图
if graph:
    plt.ion()
    plt.plot(train_history_x, train_history_loss, 'b', label='loss')
    plt.plot(train_history_x, train_history_val_loss, 'r', label='val_loss')
    plt.legend()

for i in range(epochs_total):
    history = model.fit(
        train_np_array_x,
        train_np_array_y,
        batch_size = train_dataset_length,
        epochs = epochs,
        verbose = 0,
        shuffle = False,
        validation_data = (test_np_array_x, test_np_array_y)
    )
    tf.keras.models.save_model(
        model,
        'models' + os.sep + str(i) + '.model',
        overwrite = True,
        include_optimizer = False
    )
    with open('models' + os.sep + 'models_loss.csv', 'a+', encoding='UTF8', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow([i + 1, history.history['loss'][-1], history.history['val_loss'][-1], str(i + 1) + '.model'])
    print("         id=%d,loss=%f,val_loss=%f,file=\"%s\"" % (i + 1, history.history['loss'][-1], history.history['val_loss'][-1], str(i + 1) + '.model'))
    if graph:
        train_history_x.append(i + 1)
        train_history_loss.append(history.history['loss'][-1])
        train_history_val_loss.append(history.history['val_loss'][-1])
        #plt.clf()
        plt.plot(train_history_x, train_history_loss, 'b', label='loss')
        plt.plot(train_history_x, train_history_val_loss, 'r', label='val_loss')
        plt.pause(0.02)
print('Step 6 : Train model - OK')

if graph:
    plt.ioff()
    plt.show()
