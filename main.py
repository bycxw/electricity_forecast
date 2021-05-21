# -*- coding: utf-8 -*-
# @Time:    2021/4/30
# @Author:  bycxw
# @File:    main.py
# @Description: Forecast the electricity consumption.

import os
import sys
import argparse
import pandas as pd # excel in python
import numpy as np # matlab in python
from tqdm import tqdm
import pmdarima as pm # arima
# from pyramid.arima import auto_arima

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="相关数据.xlsx")
    parser.add_argument("--save_dir", type=str, default="./result/")
    args = parser.parse_args()
    return args

def load_dataset(file):
    """
    Load dataset.
    :param file: str Path to the data file.
    :return:
    """
    df = pd.read_excel(file, engine="openpyxl", sheet_name=None) # 读入excel文件
    sheetnames = ["{}年用电数据".format(year) for year in range(2012, 2021)]
    industries = df[sheetnames[0]].iloc[:, 0].to_list() # 取第一个表格的第一列
    data = df[sheetnames[0]].iloc[:, 1:].to_numpy() # 取第一个表格的第2列至最后一列
    for sheetname in sheetnames[1:]:
        # for index, row in df[sheetname].iterrows():
        #     row = row.tolist()
        #     data[row[0]] += row[1:]
        data = np.concatenate((data, df[sheetname].iloc[:, 1:].to_numpy()), axis=-1) # 取出每个子表的数据，按时间拼接到一起
    return industries, data

def sum_chunk(x, chunk_size, axis=-1):
    shape = x.shape
    if axis < 0:
        axis += x.ndim
    shape = shape[:axis] + (-1, chunk_size) + shape[axis+1:]
    x = x.reshape(shape)
    return x.sum(axis=axis+1)

def data_transform(industries, data, granularity="month"):
    """
    Group data by month, season or year.
    :param industries:
    :param data: ndarray
    :param granularity: "month", "season" or "year"
    :return:
    """
    data_ = dict()
    assert(granularity in ["month", "season", "year"])
    if granularity == "month":
        pass
    elif granularity == "season":
        data = sum_chunk(data, 3)
    else:
        data = sum_chunk(data, 12)
    for i, industry in enumerate(industries):
        data_[industry] = data[i]
    return data_

def train_predict(train_data, pred_num, f=None):
    """
    Train a arima model for one industry.
    :param train_data:
    :return:
    """
    arima = pm.auto_arima(train_data, start_p=1, start_q=1, error_action="ignore")  # 模型训练
    print(arima.summary(), file=f) # 模型的基本信息
    print('-------------------------------------', file=f)
    print("gold, shape={}".format(train_data.shape), file=f)
    for y in train_data:
        print(y, file=f)
    preds_in_sample = arima.predict_in_sample() # 训练数据期间的用电量
    print("\npred in sample, shape={}".format(preds_in_sample.shape), file=f)
    for pred in preds_in_sample:
        print(round(pred, 2), file=f)
    preds = arima.predict(pred_num) # 预测未来用电量
    print("\npred for future(2021-2022), shape={}".format(preds.shape), file=f)
    for pred in preds:
        print(round(pred, 2), file=f)

def main(args):
    print("Load and preprocess data begin...")
    industries, data = load_dataset(args.dataset_path) # 把数据从excel表格里，以numpy 矩阵的形式放入内存
    data_month = data_transform(industries, data) # 数据处理成模型需要的形式（按月、季度、年聚合）
    data_season = data_transform(industries, data, granularity="season")
    data_year = data_transform(industries, data, granularity="year")
    print("Load and preprocess data done.")
    print("Train and predict...")
    for industry in tqdm(industries):
        with open(os.path.join(args.save_dir, "{}_month.txt".format(industry)), 'w', encoding='utf-8') as f:
            train_predict(data_month[industry], 24, f) # 训练模型，并预测未来
        with open(os.path.join(args.save_dir, "{}_season.txt".format(industry)), 'w', encoding='utf-8') as f:
            train_predict(data_season[industry], 8, f)
        with open(os.path.join(args.save_dir, "{}_year.txt".format(industry)), 'w', encoding='utf-8') as f:
            train_predict(data_year[industry], 2, f)
    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.save_dir):
        print("make dir: {}".format(args.save_dir))
        os.makedirs(args.save_dir)
    main(args)

# git github
# conda
# pip