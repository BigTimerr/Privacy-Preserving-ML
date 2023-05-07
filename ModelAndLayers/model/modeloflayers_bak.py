"""
# @Time : 2022/8/30 16:40
# @Author : ruetrash
# @File : model.py
"""
import time


class ModelOfLayers(object):
    def __init__(self):
        self.layers = []
        self.input = None
        # self.input_names = input_names
        # self.output_name = output_name

    def add(self, layer):
        self.layers.append(layer)

    def set_input(self, X):
        self.input = X

    def predict(self):
        for i, layer in enumerate(self.layers):

            if layer.name == "Conv2D":
                # print("**************************Conv2D开始啦**************************")
                # start_time = time.time()
                self.input = layer(self.input)
                # end_time = time.time()
                # print("卷积层计算时间", end_time - start_time)
                # print("**************************Conv2D结束啦**************************")
                # print()

            elif layer.name == "SecReLu":
                # print("**************************SecReLu层开始啦**************************")
                # start_time = time.time()
                self.input = layer(self.input)
                # ssf.debug(self.input)
                # end_time = time.time()
                # print("Relu层计算时间", end_time - start_time)
                # print("**************************SecReLu层结束啦**************************")
                # print()

            elif layer.name == "SecLinear":
                # print("**************************线性层开始啦**************************")
                # start_time = time.time()
                self.input = layer(self.input)
                # end_time = time.time()
                # print("Linear层计算时间", end_time - start_time)
                # print("**************************线性层结束啦**************************")
                # print()

            elif layer.name == "SecAvgPool2D":
                # print("**************************SecAvgPool2D开始啦**************************")
                # start_time = time.time()
                self.input = layer(self.input)
                # end_time = time.time()
                # print("SecAvgPool2D计算时间", end_time - start_time)
                # print("**************************SecAvgPool2D结束啦**************************")
                # print()

            elif layer.name == "SecMaxPool2D":
                # print("**************************SecMaxPool2D开始啦**************************")
                # start_time = time.time()
                self.input = layer(self.input)
                # end_time = time.time()
                # print("SecMaxPool2D计算时间", end_time - start_time)
                # print("**************************SecMaxPool2D结束啦**************************")
                # print()

            elif layer.name == "BatchNormalization":
                # print("**************************BatchNormalization开始啦**************************")
                # start_time = time.time()
                self.input = layer(self.input)
                # end_time = time.time()
                # print("计算时间", end_time - start_time)
                # print("**************************BatchNormalization结束啦**************************")
                # print()

            elif layer.name == "Flatten":
                # print("**************************flatten开始啦**************************")
                # start_time = time.time()
                self.input = layer(self.input)
                # end_time = time.time()
                # print("计算时间", end_time - start_time)
                # print("**************************flatten结束啦**************************")
                # print()
