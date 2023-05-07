"""
# @Time : 2022/8/30 16:45
# @Author : ruetrash
# @File : layers.py
"""
import math


import torch

import ProtocolOnRing.secret_sharing_vector_onring as ssv
import ProtocolOnRing.param as param
from ProtocolOnRing.secret_sharing_vector_onring import ShareV
import torch.nn.functional as F

Ring = param.Ring


class Layer(object):
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


class SecConv2d(Layer):
    def __init__(self, weight, stride=1, padding=0, bias=None, name="Conv2D"):
        super(SecConv2d, self).__init__(name=name)
        self.weight = weight
        self.kernel_shape = None
        if type(stride) is tuple:
            self.stride = stride[0]
        else:
            self.stride = stride
        if type(padding) is tuple:
            self.padding = padding[0]
        else:
            self.padding = padding
        self.name = name
        self.input = None
        self.out_shape = None
        self.bias = bias

    def get_out_shape(self, x: ShareV):
        n, img_c, img_h, img_w = x.value.shape
        kn, kc, kh, kw = self.kernel_shape
        out_h = math.ceil((img_h - self.kernel_shape[2] + 2 * self.padding) // self.stride) + 1
        out_w = math.ceil((img_w - self.kernel_shape[2] + 2 * self.padding) // self.stride) + 1

        return n, kn, out_h, out_w

    def forward(self, x):
        p = x.p
        tcp = x.tcp

        # 首先得到最终输出的形状
        self.kernel_shape = self.weight.shape
        self.out_shape = self.get_out_shape(x)

        # 对输入进来的图像进行padding操作
        padding = self.padding
        x.value = F.pad(x.value, (padding, padding, padding, padding), mode="constant", value=0)

        # kernel == weight 是由 model中提供的，现在作为演示从文件中获取
        weight = ShareV(self.weight, p, tcp)
        kN, kC, ksize, _ = self.kernel_shape

        # 首先处理图片和卷积核的形状问题 ->>>
        x.value = ssv.img2col(x.value, ksize, self.stride).transpose(1,2)
        weight.value = weight.value.reshape((kN, kC * ksize * ksize)).T

        # 将图片和卷积核送入到函数中进行计算,最后的结果加上bias
        output = (x @ weight)

        # 处理bias
        if self.bias is None:
            pass
        else:
            bias = ShareV(self.bias, p, tcp)
            output = output + bias

        output.value = output.value.transpose(1,2).reshape(self.out_shape)

        return output

    def __call__(self, x: ShareV):
        return self.forward(x)


class SecLinear(Layer):
    def __init__(self, weight, bias=0, name="SecLinear"):
        super().__init__(name)
        self.weight = ShareV(value=weight.T, p=0, tcp=None)
        self.bias = ShareV(value=bias, p=0, tcp=None)

    def forward(self, x: ShareV):
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

    def forward(self, x: ShareV):
        temp = ShareV(value=(torch.zeros(x.value.shape, dtype=torch.int64)), p=x.p, tcp=x.tcp)
        z = (x > temp) * x
        return z

    def __call__(self, x: ShareV):
        return self.forward(x)


class SecAvgPool2D(Layer):
    def __init__(self, kernel_size, stride, padding=0, name="SecAvgPool2D"):
        super().__init__(name)
        if type(kernel_size) is tuple:
            self.kernel_size = kernel_size[0]
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def get_out_shape(self, x: ShareV):
        n, img_c, img_h, img_w = x.value.shape
        out_h = math.ceil((img_h - self.kernel_size + 2 * self.padding) // self.stride) + 1
        out_w = math.ceil((img_w - self.kernel_size + 2 * self.padding) // self.stride) + 1
        return n, img_c, out_h, out_w

    def forward(self, x: ShareV):
        ksize = self.kernel_size
        stride = self.stride
        N, C, H, W = out_shape = self.get_out_shape(x)

        # 对图像进行padding处理
        padding = self.padding
        x.value = F.pad(x.value, (padding, padding, padding, padding), mode="constant", value=0)
        #
        z = F.avg_pool2d(x.value, self.kernel_size, self.stride)
        res = ShareV(value=z.transpose(0, 1), p=x.p, tcp=x.tcp, device=x.device)

        return res

    def __call__(self, x: ShareV):
        return self.forward(x)


class SecMaxPool2D(Layer):
    def __init__(self, kernel_size, stride, padding=0, name="SecMaxPool2D"):
        super().__init__(name)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def get_out_shape(self, x: ShareV):
        n, img_c, img_h, img_w = x.value.shape
        out_h = math.ceil((img_h - self.kernel_size + 2 * self.padding) // self.stride) + 1
        out_w = math.ceil((img_w - self.kernel_size + 2 * self.padding) // self.stride) + 1

        return (n, img_c, out_h, out_w)

    def sec_max(self, z, p, tcp):
        def max_(z):
            if z.shape[1] == 1:
                return z
            if z.shape[1] % 2 == 1:
                z_ = z[:, -1:, :]
                z = torch.cat((z, z_), dim=1)
            z0 = ShareV(value=z[:, 0::2, :], p=p, tcp=tcp)
            z1 = ShareV(value=z[:, 1::2, :], p=p, tcp=tcp)

            b0 = (z0 >= z1)
            b1 = (z1 > z0)

            b0 = b0 * z0
            b1 = b1 * z1

            return (b0.value + b1.value) % (Ring * 2)

        if z.shape[1] == 1:
            return z
        else:
            z = max_(z)
        return self.sec_max(z, p, tcp)

    def forward(self, x: ShareV):
        N, C, W, H = x.value.shape
        ksize = self.kernel_size
        stride = self.stride
        out_shape = self.get_out_shape(x)

        # 首先是padding操作
        padding = self.padding
        x.value = F.pad(x.value, (padding, padding, padding, padding), mode="constant", value=0)

        x.value = ssv.img2col(x.value, ksize, stride)
        xs = []
        for i in range(0, C):
            xs.append(self.sec_max(x.value[:, i * ksize * ksize:(i + 1) * ksize * ksize, :], x.p, x.tcp))
        xs = torch.cat(xs, dim=1).reshape(out_shape).to()

        return ShareV(xs, x.p, x.tcp)

    def __call__(self, x: ShareV):
        return self.forward(x)


# Y = (X - moving_mean) / sqrt(moving_variance + eps) * gamma + beta
class BatchNormalization(Layer):  # y = x * gamma + beta
    def __init__(self, gamma, beta, name="BatchNormalization"):
        super().__init__(name)
        self.moving_mean = None
        self.moving_variance = None
        self.eps = None
        self.gamma = gamma
        self.beta = beta

    def forward(self, x: ShareV):
        gamma = ShareV(self.gamma, x.p, x.tcp)
        beta = ShareV(self.beta, x.p, x.tcp)
        res = (x * gamma) + beta
        return res

    def __call__(self, x: ShareV):
        return self.forward(x)


class Flatten(Layer):
    def __init__(self, name="Flatten"):
        super().__init__(name)

    def output_shape(self, x):
        n, c, w, h = x.value.shape
        return (n, c * w * h)

    def __call__(self, x: ShareV):
        out_shape = self.output_shape(x)
        x.value = x.value.reshape(out_shape)
        return x
