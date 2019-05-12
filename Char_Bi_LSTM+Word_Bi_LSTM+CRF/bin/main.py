# -*- coding: utf-8 -*-
# @Time    : 2019/5/10 11:22
# @Author  : Weiyang
# @File    : main.py.py
'''
训练模型，然后预测
'''

from src.TrainModel import TrainModel
from src.ModelPredict import ModelPredict

def main():
    # 训练模型
    model = TrainModel()
    model.train()
    # 加载训练完毕的模型
    mp = ModelPredict()
    LINE = 'John lives in New York'
    mp.predict(LINE)

if __name__ == '__main__':
    main()
