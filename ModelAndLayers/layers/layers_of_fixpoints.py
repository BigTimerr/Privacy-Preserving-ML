"""
# @Time : 2022/10/17 16:24
# @Author : ruetrash
# @File : layers_of_fixpoints.py
"""
import math
import torch
import ProtocolOnRing.secret_sharing_fixpoint as ssf
from ProtocolOnRing.secret_sharing_fixpoint import ShareFloat
import ProtocolOnRing.param as param
import torch.nn.functional as F

Q = param.Q
BASE = param.BASE
scaled = param.scaled


class Layer(object):
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


class SecConv2d(Layer):
    def __init__(self, weight, stride=(1, 1), padding=(0, 0), bias=None, name="Conv2D"):
        super(SecConv2d, self).__init__(name=name)
        self.weight = weight
        self.kernel_shape = None
        if type(stride) is tuple:
            self.stride = stride[0]
        else:
            self.stride = stride
        if type(padding) is tuple:
            padding = padding[0]
        else:
            padding = padding
        self.padding = padding
        self.name = name
        self.input = None
        self.out_shape = None
        self.bias = bias

    def get_out_shape(self, x: ShareFloat):
        n, img_c, img_h, img_w = x.value.shape
        kn, kc, kh, kw = self.kernel_shape
        out_h = math.ceil((img_h - self.kernel_shape[2] + 2 * self.padding) // self.stride) + 1
        out_w = math.ceil((img_w - self.kernel_shape[2] + 2 * self.padding) // self.stride) + 1

        return n, kn, out_h, out_w

    def forward(self, x: ShareFloat):
        p = x.p
        tcp = x.tcp

        # 首先得到最终输出的形状
        self.kernel_shape = self.weight.shape
        self.out_shape = self.get_out_shape(x)

        # 对输入进来的图像进行padding操作
        padding = self.padding
        x.value = F.pad(x.value, (padding, padding, padding, padding), mode="constant", value=0)

        weight = ShareFloat(self.weight, p, tcp)
        kN, kC, ksize, _ = self.kernel_shape

        # 首先处理图片和卷积核的形状问题 ->>>
        x.value = ssf.img2col(x.value, ksize, self.stride).transpose(1, 2)
        weight.value = weight.value.reshape((kN, kC * ksize * ksize)).T

        # 将图片和卷积核送入到函数中进行计算,最后的结果加上bias
        output = (x @ weight)

        # 处理bias
        if self.bias is None:
            pass
        else:
            bias = ShareFloat(self.bias, p, tcp)
            output = output + bias

        output.value = (output.value.transpose(1, 2)).reshape(self.out_shape)

        return output

    def __call__(self, x: ShareFloat):
        return self.forward(x)


class SecLinear(Layer):
    def __init__(self, weight, bias, name="SecLinear"):
        super().__init__(name)
        self.weight = ShareFloat(value=weight.T, p=0, tcp=None)
        self.bias = ShareFloat(value=bias, p=0, tcp=None)

    def forward(self, x: ShareFloat):
        self.weight.p = self.bias.p = x.p
        self.weight.tcp = self.bias.tcp = x.p
        x.value = x.value.reshape(x.value.shape[0], -1)
        z = (x @ self.weight) + self.bias
        return z

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class SecReLu(Layer):
    def __init__(self, name="SecReLu"):
        super().__init__(name)

    def forward(self, x: ShareFloat):
        temp = ShareFloat(value=(torch.zeros(x.value.shape, dtype=torch.int64)), p=x.p, tcp=x.tcp)
        z = (x > temp) * scaled
        z = z * x
        return z

    def __call__(self, x: ShareFloat):
        return self.forward(x)


class SecAvgPool2D(Layer):
    def __init__(self, kernel_shape, stride, padding=(0, 0), name="SecAvgPool2D"):
        super().__init__(name)
        if type(kernel_shape) is tuple:
            self.kernel_shape = kernel_shape[0]
        else:
            self.kernel_shape = kernel_shape
        self.stride = stride
        if type(padding) is tuple:
            padding = padding[0]
        else:
            padding = padding
        self.padding = padding

    def get_out_shape(self, x: ShareFloat):
        n, img_c, img_h, img_w = x.value.shape
        out_h = math.ceil((img_h - self.kernel_shape + 2 * self.padding) // self.stride) + 1
        out_w = math.ceil((img_w - self.kernel_shape + 2 * self.padding) // self.stride) + 1
        return n, img_c, out_h, out_w

    def forward(self, x: ShareFloat):
        # 对图像进行padding处理
        padding = self.padding
        x.value = F.pad(x.value, (padding, padding, padding, padding), mode="constant", value=0)

        z = F.avg_pool2d(x.value, self.kernel_shape, self.stride)
        res = ShareFloat(value=z.transpose(0, 1), p=x.p, tcp=x.tcp)

        return res

    def __call__(self, x: ShareFloat):
        return self.forward(x)


class SecMaxPool2D(Layer):
    def __init__(self, kernel_shape, stride, padding=(0, 0), name="SecMaxPool2D"):
        super().__init__(name)
        self.kernel_shape = kernel_shape
        self.stride = stride
        if type(padding) is tuple:
            padding = padding[0]
        else:
            padding = padding
        self.padding = padding

    def get_out_shape(self, x: ShareFloat):
        n, img_c, img_h, img_w = x.value.shape
        out_h = math.ceil((img_h - self.kernel_shape + 2 * self.padding) // self.stride) + 1
        out_w = math.ceil((img_w - self.kernel_shape + 2 * self.padding) // self.stride) + 1

        return (n, img_c, out_h, out_w)

    def sec_max(self, z, p, tcp):
        def max_(z):
            if z.shape[1] == 1:
                return z
            if z.shape[1] % 2 == 1:
                z_ = z[:, -1:, :]
                z = torch.cat((z, z_), dim=1)
            z0 = ShareFloat(value=z[:, 0::2, :], p=p, tcp=tcp)
            z1 = ShareFloat(value=z[:, 1::2, :], p=p, tcp=tcp)

            b0 = (z0 >= z1)
            b1 = (z1 > z0)

            b0.value = b0.value * scaled
            b1.value = b1.value * scaled

            b0 = b0 * z0
            b1 = b1 * z1

            return (b0.value + b1.value) % Q

        if z.shape[1] == 1:
            return z
        else:
            z = max_(z)
        return self.sec_max(z, p, tcp)

    def forward(self, x: ShareFloat):
        N, C, W, H = x.value.shape
        k_size = self.kernel_shape
        stride = self.stride
        out_shape = self.get_out_shape(x)

        # 首先是padding操作
        padding = self.padding
        x.value = F.pad(x.value, (padding, padding, padding, padding), mode="constant", value=0)

        x.value = ssf.img2col(x.value, k_size, stride)
        xs = []
        for i in range(0, C):
            xs.append(self.sec_max(x.value[:, i * k_size * k_size:(i + 1) * k_size * k_size, :], x.p, x.tcp))
            # temp = temp.reshape(out_shape[0], out_shape[2], out_shape[3])
            # xs.append(temp)

        xs = torch.cat(xs, dim=1).reshape(out_shape)

        return ShareFloat(xs, x.p, x.tcp)

    def __call__(self, x: ShareFloat):
        return self.forward(x)


class Flatten(Layer):
    def __init__(self, name="Flatten"):
        super().__init__(name)

    def output_shape(self, x):
        n, c, w, h = x.value.shape
        return (n, c * w * h)

    def __call__(self, x: ShareFloat):
        out_shape = self.output_shape(x)
        x.value = x.value.reshape(out_shape)
        return x
