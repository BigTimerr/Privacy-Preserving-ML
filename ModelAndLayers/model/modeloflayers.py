"""
# @Time : 2022/8/30 16:40
# @Author : ruetrash
# @File : model.py
"""
import time
from ProtocolOnRing import secret_sharing_fixpoint as ssf

class ModelOfLayers(object):
    def __init__(self, input_names=None, output_name=None):
        self.layers = []
        self.input = None
        self.input_names = input_names
        self.output_name = output_name
        self.output = {}

    def add(self, layer):
        self.layers.append(layer)

    def set_input(self, X):
        self.input = X
        self.output["input"] = X

    def predict(self):
        for i, layer in enumerate(self.layers):

            if layer.name == "Conv2D":
                # print("==================分割线====================")
                output = layer(self.output[layer.input_name])
                self.output[layer.output_name] = output
                # print("Conv2D")
                # ssf.debug(output)

            elif layer.name == "SecReLu":
                output = layer(self.output[layer.input_name])
                self.output[layer.output_name] = output
                # print("SecReLu")
                # ssf.debug(output)

            elif layer.name == "SecLinear":
                output = layer(self.output[layer.input_name])
                self.output[layer.output_name] = output

            elif layer.name == "SecAvgPool2D":
                output = layer(self.output[layer.input_name])
                self.output[layer.output_name] = output

            elif layer.name == "SecMaxPool2D":
                output = layer(self.output[layer.input_name])
                self.output[layer.output_name] = output
                # print("SecMaxPool2D")
                # ssf.debug(output)

            elif layer.name == "BatchNormalization":
                output = layer(self.output[layer.input_name])
                self.output[layer.output_name] = output

            elif layer.name == "Flatten":
                output = layer(self.output[layer.input_name])
                self.output[layer.output_name] = output

            elif layer.name == "SecADD":
                output = layer(self.output[layer.input_name[0]], self.output[layer.input_name[1]])
                self.output[layer.output_name] = output

            elif layer.name == "SecMatMul":
                output = layer(self.output[layer.input_name[0]], self.output[layer.input_name[1]])
                self.output[layer.output_name] = output

            elif layer.name == "SecTranspose":
                output = layer()
                self.output[layer.output_name] = output

            elif layer.name == "SecReshape":
                output = layer(self.output[layer.input_name[0]], self.output[layer.input_name[1]])
                self.output[layer.output_name] = output

            elif layer.name == "SecGemm":
                output = layer(self.output[layer.input_name])
                self.output[layer.output_name] = output

            elif layer.name == "SecPad":
                output = layer(self.output[layer.input_name[0]], self.output[layer.input_name[1]])
                self.output[layer.output_name] = output

            elif layer.name == "SecUnsqueeze":
                output = layer(self.output[layer.input_name])
                self.output[layer.output_name] = output

            elif layer.name == "SecConcat":
                output = layer(self.output[layer.input_name[0]], self.output[layer.input_name[1]])
                self.output[layer.output_name] = output

            elif layer.name == "SecBN2d":
                output = layer(self.output[layer.input_name])
                self.output[layer.output_name] = output
                # print("SecBN2d")
                # ssf.debug(output)

