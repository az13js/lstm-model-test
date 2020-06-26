# -*- coding: UTF-8 -*-

# 出于通用性的考虑，最后生成的文件是没有表格头部的，里面的东西都是纯数字，而且每一行
# 的数据项目数量都是一样的

import csv
import math

results = []

line = 1
with open('dataset.csv', newline='', encoding='UTF-8') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        if line > 1:
            results.append(row)
        line = line + 1
        #if line % 100 == 0:
        #    print(line)

print('File "dataset_no_title.csv"')
with open('dataset_no_title.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(results)

results_diff = []
results_length = len(results)
print('data length:')
print(results_length)

diff_item_total = results_length - 1

for i in range(diff_item_total):
    row_a = results[i]
    row_b = results[i + 1]
    row_diff = row_b.copy()
    row_diff[0] = i + 1
    row_diff[2] = math.log(float(row_b[2]) / float(row_a[2]))
    row_diff[3] = math.log(float(row_b[3]) / float(row_a[3]))
    row_diff[4] = math.log(float(row_b[4]) / float(row_a[4]))
    row_diff[5] = math.log(float(row_b[5]) / float(row_a[5]))
    row_diff[6] = math.log(float(row_b[6]) / float(row_a[6]))
    row_diff[7] = math.log(float(row_b[7]) / float(row_a[7]))
    results_diff.append(row_diff)

# results 不再需要了
results = []

print('File "dataset_diff.csv"')
with open('dataset_diff.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(results_diff)

print('length of dataset_diff.csv')
print(len(results_diff))

the_end = []
for i in range(diff_item_total):
    item = 0
    row = []
    for data in results_diff[i]:
        if 0 != item and 1 != item:
            row.append(data)
        item = item + 1
    the_end.append(row)

print('File "dataset_the_end.csv"')
with open('dataset_the_end.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(the_end)
