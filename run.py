# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
import csv
import os

# 为了方便起见，这里把可以配置的参数在开头进行声明
# 模型文件
file_name = 'Final.model'
# 输入数据文件
data_input_file = 'dataset_the_end.csv'
model_sequence_length = 30

# 处理输入数据的方法
def input_data_processor(input_data_row):
    return input_data_row

all_data_from_csv = []
# 从csv文件中读取训练用的数据，顺序
print('Step 1 : Read data from dataset_the_end.csv')
line_num = 1
with open(data_input_file, newline='') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        data_row = []
        for item in row:
            data_row.append(float(item))
        all_data_from_csv.append(data_row)
        line_num = line_num + 1
print('Step 1 : Read data from dataset_the_end.csv - OK.')

length_of_all_data_from_csv = len(all_data_from_csv)
if length_of_all_data_from_csv < model_sequence_length:
    print('insufficient')
    print(length_of_all_data_from_csv)
    exit()

print('Step 2 : Build data for model')
# 可以用来给模型作为输入数据使用的数据
all_data_for_model_inputs = []
length_of_all_data_for_model_inputs = length_of_all_data_from_csv - model_sequence_length + 1
for i in range(length_of_all_data_for_model_inputs):
    input_data = []
    for j in range(model_sequence_length):
        input_data.append(input_data_processor(all_data_from_csv[i + j].copy()))
    all_data_for_model_inputs.append(input_data)
print('Step 2 : Build data for model - OK')

print('Step 3 : Load and predict')
model = tf.keras.models.load_model(file_name)
model.summary()

predict_result = model.predict(all_data_for_model_inputs)

with open('models' + os.sep + 'predict.csv', 'a+', encoding='UTF8', newline='') as f:
    csv_write = csv.writer(f)
    for row in predict_result:
        csv_write.writerow(row)

print('Step 3 : Load and predict - OK')
