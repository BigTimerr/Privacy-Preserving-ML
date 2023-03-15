# **[Privacy-Preserving-ML](https://gitee.com/ruetrash/Privacy-Preserving-ML)**

## 背景

Privacy-Preserving-ML是一个建立在PyTorch上的隐私保护机器学习框架。它的目标是使机器学习从业人员可以简便的使用隐私保护计算技术。目前，它将安全的多方计算作为其安全的计算后端，并为ML研究人员提供了三个主要好处：

- 实现了基础计算协议，包括安全加法、安全乘法、安全比较、安全矩阵乘法、安全MSB协议。
- 实现了神经网络协议，安全卷积、安全池化、安全全连接、安全ReLU函数。
- 基于上述协议，已经实现了ResNet（**正在开发中**）、AlexNet、VGGNet等网络模型的隐私保护预测过程。并支持将训练后的明文模型自动部署到双云模型下。

本框架目前还没有投入生产，主要用途是作为研究框架。

## 安装

本项目需要Pytorch>=1.8.0支持，下面演示如何通过conda安装本项目所需环境

```
conda create -n pytorch python=3.9
conda activate pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

## 使用说明

- [BasicTest](https://github.com/BigTimerr/Privacy-Preserving-ML/tree/main/BasicTest)

  BasicTest包中包含本项目的所有基础计算协议（整数部分）的测试代码。
- [FixPointTest](https://github.com/BigTimerr/Privacy-Preserving-ML/tree/main/FixPointTest)

  FixPointText包中包含本项目的所有基础计算协议（点点数表示小数部分）的测试代码。
- [ModelAndLayers](https://github.com/BigTimerr/Privacy-Preserving-ML/tree/main/ModelAndLayers)

  ModelAndLayers包含本项目中所有机器学习部分的代码。
- [MSB](https://github.com/BigTimerr/Privacy-Preserving-ML/tree/main/MSB)

  MSB包含本项目中安全MSB协议的实现代码以及测试代码。
- [ProtocolOnRing](https://github.com/BigTimerr/Privacy-Preserving-ML/tree/main/ProtocolOnRing)

  ProtocolOnRing是本项目的核心代码。
- [TCP](https://github.com/BigTimerr/Privacy-Preserving-ML/tree/main/TCP)

  TCP是包含本项目信息传输所使用到的代码以及测试代码。

## 示例

运行所有测试代码均需要在项目根目录下启动，如下面所展示的示例。

在启动测试之前，需要运行  [triples.py](ProtocolOnRing\triples.py) 和 [msb_triples_vector.py](MSB\msb_triples_vector.py) 文件生成程序所用到的乘法三元组

```
# 开启两个终端，分别输入以下代码
python BasicTest/vector_test_client_onring.py
python BasicTest/vector_test_server_onring.py
```

## 维护者

本项目由西安电子科技大学NSS小组负责维护。

## 使用许可

Privacy-Preserving-ML 基于 GPL3.0 ，如在[LICENSE](https://gitee.com/ruetrash/Privacy-Preserving-ML/blob/main/LICENSE)所述。
