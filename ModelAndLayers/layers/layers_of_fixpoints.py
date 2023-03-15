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

Q = param.Ring
BASE = param.BASE
scaled = param.scaled


class Layer(object):
    def __init__(self, input_name=None, output_name=None, name=None):
        self.input_name = input_name
        self.output_name = output_name
        self.name = name

    def get_name(self):
        return self.name

    def set_input_and_output(self, input_name, output_name):
        self.input_name = input_name
        self.output_name = output_name


class SecConv2d(Layer):
    def __init__(self, weight, stride=(1, 1), padding=(0, 0), bias=None, device="cpu", name="Conv2D"):
        super(SecConv2d, self).__init__(name=name)
        self.weight = weight
        self.kernel_shape = None
        if type(stride) in (tuple, list):
            self.stride = stride[0]
        else:
            self.stride = stride
        if type(padding) in (tuple, list):
            padding = padding[0]
        else:
            padding = padding
        self.padding = padding
        self.name = name
        self.input = None
        self.out_shape = None
        self.bias = bias
        self.device = device
        if self.device == "cuda":
            self.weight = self.weight.to(device)

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

        weight_ = ShareFloat(self.weight, p, tcp, self.device)
        weight = ShareFloat(weight_.value, p, tcp, self.device)
        kN, kC, ksize, _ = self.kernel_shape

        # 首先处理图片和卷积核的形状问题 ->>>
        x.value = ssf.img2col(x.value, ksize, self.stride, self.device).transpose(1, 2)
        weight.value = weight.value.reshape((kN, kC * ksize * ksize)).T

        # 将图片和卷积核送入到函数中进行计算,最后的结果加上bias
        output = (x @ weight)

        # 处理bias
        if self.bias is None:
            pass
        else:
            bias = ShareFloat(self.bias, p, tcp, device=self.device)
            output = output + bias

        output.value = (output.value.transpose(1, 2)).reshape(self.out_shape)

        return output

    def __call__(self, x: ShareFloat):
        x_ = ShareFloat(x.value, x.p, x.tcp, device=x.device)
        return self.forward(x_)


class SecLinear(Layer):
    def __init__(self, weight, bias, name="SecLinear"):
        super(SecLinear, self).__init__(name=name)
        self.weight = ShareFloat(value=weight.T, p=0, tcp=None, device=None)
        self.bias = ShareFloat(value=bias, p=0, tcp=None, device=None)

    def forward(self, x: ShareFloat):
        self.weight.p = self.bias.p = x.p
        self.weight.tcp = self.bias.tcp = x.p
        self.weight.device = self.bias.device = x.device
        x.value = x.value.reshape(x.value.shape[0], -1)
        z = (x @ self.weight) + self.bias
        return z

    def __call__(self, x: ShareFloat):
        x_ = ShareFloat(x.value, x.p, x.tcp, x.device)
        return x_


class SecReLu(Layer):
    def __init__(self, name="SecReLu"):
        super(SecReLu, self).__init__(name=name)

    def forward(self, x: ShareFloat):
        temp = ShareFloat(value=(torch.zeros(x.value.shape, dtype=torch.long)), p=x.p, tcp=x.tcp, device=x.device)
        z = (x > temp) * scaled
        z = z * x
        return z

    def __call__(self, x: ShareFloat):
        x_ = ShareFloat(x.value, x.p, x.tcp, x.device)
        return self.forward(x_)


class SecAvgPool2D(Layer):
    def __init__(self, kernel_shape, stride, padding=(0, 0), name="SecAvgPool2D"):
        super(SecAvgPool2D, self).__init__(name=name)
        if type(kernel_shape) in (tuple, list):
            self.kernel_shape = kernel_shape[0]
        else:
            self.kernel_shape = kernel_shape
        if type(stride) in (tuple, list):
            stride = stride[0]
        else:
            stride = stride
        self.stride = stride
        if type(padding) in (tuple, list):
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
        res = ShareFloat(value=z.transpose(0, 1), p=x.p, tcp=x.tcp, device=x.device)

        return res

    def __call__(self, x: ShareFloat):
        x_ = ShareFloat(x.value, x.p, x.tcp, x.device)
        return self.forward(x_)


class SecMaxPool2D(Layer):
    def __init__(self, kernel_shape, stride, padding=(0, 0), device="cpu", name="SecMaxPool2D"):
        super(SecMaxPool2D, self).__init__(name=name)
        if type(kernel_shape) in (tuple, list):
            self.kernel_shape = kernel_shape[0]
        else:
            self.kernel_shape = kernel_shape
        if type(stride) in (tuple, list):
            stride = stride[0]
        else:
            stride = stride
        self.stride = stride
        if type(padding) in (tuple, list):
            padding = padding[0]
        else:
            padding = padding
        self.padding = padding
        self.device = device

    def get_out_shape(self, x: ShareFloat):
        n, img_c, img_h, img_w = x.value.shape
        out_h = math.ceil((img_h - self.kernel_shape + 2 * self.padding) // self.stride) + 1
        out_w = math.ceil((img_w - self.kernel_shape + 2 * self.padding) // self.stride) + 1

        return (n, img_c, out_h, out_w)

    def sec_max(self, z, p, tcp, device):
        def max_(z):
            if z.shape[1] == 1:
                return z
            if z.shape[1] % 2 == 1:
                z_ = z[:, -1:, :]
                z = torch.cat((z, z_), dim=1)
            z0 = ShareFloat(value=z[:, 0::2, :], p=p, tcp=tcp, device=device)
            z1 = ShareFloat(value=z[:, 1::2, :], p=p, tcp=tcp, device=device)

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
        return self.sec_max(z, p, tcp, device)

    def forward(self, x: ShareFloat):
        N, C, W, H = x.value.shape
        k_size = self.kernel_shape
        stride = self.stride
        out_shape = self.get_out_shape(x)

        # 首先是padding操作
        padding = self.padding
        x.value = F.pad(x.value, (padding, padding, padding, padding), mode="constant", value=0)

        x.value = ssf.img2col(x.value, k_size, stride, device=self.device)
        xs = []
        for i in range(0, C):
            xs.append(self.sec_max(x.value[:, i * k_size * k_size:(i + 1) * k_size * k_size, :], x.p, x.tcp, x.device))
            # temp = temp.reshape(out_shape[0], out_shape[2], out_shape[3])
            # xs.append(temp)

        xs = torch.cat(xs, dim=1).reshape(out_shape).to(self.device)

        return ShareFloat(xs, x.p, x.tcp, x.device)

    def __call__(self, x: ShareFloat):
        x_ = ShareFloat(x.value, x.p, x.tcp, x.device)
        return self.forward(x_)


class Flatten(Layer):
    def __init__(self, name="Flatten"):
        super(Flatten, self).__init__(name=name)

    def output_shape(self, x):
        n, c, w, h = x.value.shape
        return (n, c * w * h)

    def __call__(self, x: ShareFloat):
        out_shape = self.output_shape(x)
        x.value = x.value.reshape(out_shape)
        return x


class SecMatMul(Layer):
    def __init__(self, name="SecMatMul"):
        super(SecMatMul, self).__init__(name=name)

    def __call__(self, x: ShareFloat, y: ShareFloat):
        return x @ y


class SecADD(Layer):
    def __init__(self, name="SecADD"):
        super(SecADD, self).__init__(name=name)

    def __call__(self, x: ShareFloat, y: ShareFloat):
        z = x + y
        return z


class SecTranspose(Layer):
    def __init__(self, weight, name="SecTranspose"):
        super(SecTranspose, self).__init__(name=name)
        self.weight = weight

    def __call__(self):
        y = self.weight.value.T
        return ShareFloat(y, self.weight.p, self.weight.tcp, self.weight.device)


class SecReshape(Layer):
    def __init__(self, name="SecReshape"):
        super(SecReshape, self).__init__(name=name)

    def __call__(self, x: ShareFloat, shape):
        y = x.value.reshape(tuple(shape.value.tolist()))
        return ShareFloat(y, x.p, x.tcp, x.device)


class SecGemm(Layer):
    def __init__(self, weight, bias, device, name="SecGemm"):
        super(SecGemm, self).__init__(name=name)
        self.device = device
        self.weight = ShareFloat(value=weight.T, p=0, tcp=None, device=device)
        self.bias = ShareFloat(value=bias, p=0, tcp=None, device=device)

    def forward(self, x):
        self.weight.p = self.bias.p = x.p
        self.weight.tcp = self.bias.tcp = x.tcp
        z = (x @ self.weight) + self.bias
        return z

    def __call__(self, x: ShareFloat):
        return self.forward(x)


class SecPad(Layer):
    def __init__(self, mode, value=0, name="SecPad"):
        super(SecPad, self).__init__(name=name)
        self.mode = mode
        self.value = value

    def forward(self, x, pad):
        y = torch.nn.functional.pad(x.value, pad.value.tolist(), self.mode, self.value)
        return ShareFloat(y, x.p, x.tcp, x.device)

    def __call__(self, x: ShareFloat, pad):
        return self.forward(x, pad)


class SecUnsqueeze(Layer):
    def __init__(self, dim, name="SecUnsqueeze"):
        super(SecUnsqueeze, self).__init__(name=name)
        self.dim = dim

    def forward(self, x):
        y = torch.unsqueeze(x.value, self.dim.value.item())
        return ShareFloat(y, x.p, x.tcp, x.device)

    def __call__(self, x: ShareFloat):
        return self.forward(x)


class SecConcat(Layer):
    def __init__(self, axis, name="SecConcat"):
        super(SecConcat, self).__init__(name=name)
        self.axis = axis

    def forward(self, x, y):
        z = torch.cat((x.value, y.value), dim=self.axis)
        return ShareFloat(z, x.p, x.tcp, x.device)

    def __call__(self, x: ShareFloat, y: ShareFloat):
        return self.forward(x, y)


class SecCopy(Layer):
    def __init__(self, name="SecCopy"):
        super().__init__(name=name)

    def forward(self, x):
        y = ShareFloat(x.value, x.p, x.tcp)
        return y

    def __call__(self, x: ShareFloat):
        return self.forward(x)


class SecBN2d(Layer):
    def __init__(self, gamma, beta, running_mean, running_var, epsilon, name="SecBN2d"):
        super().__init__(name=name)
        self.gamma = gamma
        self.beta = beta
        self.running_mean = running_mean
        self.running_var = running_var
        self.epsilon = epsilon

    def forward(self, x: ShareFloat):  # # Y = (X - running_mean) / sqrt(running_var + eps) * gamma + beta # #

        self.running_mean.value = self.running_mean.value.unsqueeze(1)
        self.running_mean.value = self.running_mean.value.unsqueeze(2)
        self.running_mean.value = self.running_mean.value.repeat(1, x.value.shape[2], x.value.shape[3])

        self.running_var = self.running_var.unsqueeze(1)
        self.running_var = self.running_var.unsqueeze(2)

        ssf.debug(x)

        x = x - self.running_mean

        divide = torch.floor(self.running_var + self.epsilon).long()
        print(divide)
        print(x.value)
        # x.value = torch.floor(x.value / (self.running_var + self.epsilon))
        x.value = torch.floor(torch.true_divide(x.value, divide).long())
        print(x.value)
        ssf.debug(x)

        self.gamma.value = self.gamma.value.unsqueeze(1)
        self.gamma.value = self.gamma.value.unsqueeze(2)

        self.beta.value = self.beta.value.unsqueeze(1)
        self.beta.value = self.beta.value.unsqueeze(2)

        x = x * self.gamma
        x = x + self.beta

        return x

    def __call__(self, x: ShareFloat):
        x_ = ShareFloat(x.value, x.p, x.tcp, x.device)
        return self.forward(x_)
