"""
# @Time : 2022/12/19 16:13
# @Author : ruetrash
# @File : onnx_converter.py
"""
import io
import onnx
import torch
import torch.nn as nn
from torch.onnx import OperatorExportTypes
from onnx import numpy_helper
from ModelAndLayers.model import modeloflayers
from ModelAndLayers.layers import layers_of_fixpoints as layers
import ProtocolOnRing.secret_sharing_fixpoint as ssf
from torch.onnx import TrainingMode


def from_pytorch(pytorch_model, dummy_input, party, device):
    """
    利用dummy_input 把 pytorch_model 转化为框架所支持的model。
    :param pytorch_model:
    :param dummy_input:
    :return:
    """

    kwargs = {
        "do_constant_folding": False,
        "export_params": True,
        "input_names": ["input"],
        "output_names": ["output"],
        "operator_export_type": OperatorExportTypes.ONNX,
        "opset_version": 17,
        "training": TrainingMode.EVAL
    }

    # 返回 pytorch_model 的包含onnx graph 的io流
    f = io.BytesIO()
    torch.onnx.export(pytorch_model, dummy_input, f, **kwargs)
    f.seek(0)

    # 将io流转为onnx模型
    onnx_model = _load_onnx_model(f)

    # 将onnx模型转为框架所支持的model
    secure_model = _to_secure_model(onnx_model, party, device)

    return secure_model


def _to_secure_model(onnx_model, party, device):
    """
    Function that converts an `onnx_model` to a CrypTen model.
    """

    # print(onnx_model)

    # 首先得到整个模型的input_names， output_names，并且init一个secure_model
    input_names, output_names = _get_input_output_names(onnx_model)
    assert len(output_names) == 1, "Only one output per model supported."
    secure_model = modeloflayers.ModelOfLayers(str(input_names), str(output_names[0]))

    # 把模型中的参数全部保存在dict中,只有client需要做这一步
    parameters = {}
    if party.party == 1:
        for node in onnx_model.graph.initializer:
            param = torch.from_numpy(numpy_helper.to_array(node))
            parameters[node.name] = param

    # 顺序遍历onnx模型的所有node节点
    if party.party == 1:
        for node in onnx_model.graph.node:
            # get attributes and node type:
            attributes = {attr.name: _get_attribute_value(attr) for attr in node.attribute}

            if node.op_type == "Identity":
                value = parameters[node.input[0]]
                party.send_torch_array(value)
                secure_model.output[node.output[0]] = ssf.ShareFloat(value, party.party, party, device)

            elif node.op_type == "Conv":
                output_name = node.output[0]

                pads = attributes["pads"]
                strides = attributes["strides"]

                w0, w1 = ssf.share_float(ssf.encode(parameters[node.input[1]]))
                party.send_torch_array(w0)

                if len(node.input) == 2:
                    b0 = b1 = None
                else:
                    if node.input[2] in parameters:
                        b0, b1 = ssf.share_float(ssf.encode(parameters[node.input[2]]))
                        party.send_torch_array(b0)
                    else:
                        b0, b1 = ssf.share_float(ssf.encode(secure_model.output[node.input[2]].value))
                        party.send_torch_array(b0)

                layer = layers.SecConv2d(weight=w1, stride=strides, padding=pads, bias=b1, device=device)
                layer.set_input_and_output(node.input[0], output_name)
                secure_model.add(layer)

            elif node.op_type == "Relu":

                layer = layers.SecReLu()
                layer.set_input_and_output(node.input[0], node.output[0])
                secure_model.add(layer)

            elif node.op_type == "MaxPool":
                pads = attributes["pads"]
                strides = attributes["strides"]
                kernel_shape = attributes["kernel_shape"]

                layer = layers.SecMaxPool2D(kernel_shape, strides, pads, device)
                layer.set_input_and_output(node.input[0], node.output[0])
                secure_model.add(layer)

            elif node.op_type == "AveragePool":
                pads = attributes["pads"]
                strides = attributes["strides"]
                kernel_shape = attributes["kernel_shape"]

                layer = layers.SecAvgPool2D(kernel_shape, strides, pads)
                layer.set_input_and_output(node.input[0], node.output[0])
                secure_model.add(layer)

            elif node.op_type == "MatMul":
                layer = layers.SecMatMul()
                layer.set_input_and_output((node.input[0], node.input[1]), node.output[0])
                secure_model.add(layer)

            elif node.op_type == "Gemm":
                if node.input[1] in parameters:
                    w0, w1 = ssf.share_float(ssf.encode(parameters[node.input[1]]))
                else:
                    w0, w1 = ssf.share_float(ssf.encode(secure_model.output[node.input[1]].value))
                if node.input[2] in parameters:
                    b0, b1 = ssf.share_float(ssf.encode(parameters[node.input[2]]))
                else:
                    b0, b1 = ssf.share_float(ssf.encode(secure_model.output[node.input[2]].value))
                party.send_torch_array(w0)
                party.send_torch_array(b0)

                layer = layers.SecGemm(w1, b1, device)
                layer.set_input_and_output(node.input[0], node.output[0])
                secure_model.add(layer)

            elif node.op_type == "Add":

                # 如果加法的一方是linear层的bias，需要特殊处理
                if node.input[0].find("bias") != -1:
                    bias = parameters[node.input[0]]
                    b0, b1 = ssf.share_float(ssf.encode(bias))
                    party.send_torch_array(b0)

                    secure_model.output[node.input[0]] = ssf.ShareFloat(b1, party.party, party, device)

                layer = layers.SecADD()
                layer.set_input_and_output((node.input[0], node.input[1]), node.output[0])
                secure_model.add(layer)

            elif node.op_type == "Transpose":
                weight = parameters[node.input[0]]

                w0, w1 = ssf.share_float(ssf.encode(weight))
                party.send_torch_array(w0)

                layer = layers.SecTranspose(ssf.ShareFloat(w1, party.party, party, device), device)
                layer.set_input_and_output(node.input[0], node.output[0])
                secure_model.add(layer)

            elif node.op_type == "Constant":  # 这个节点的value 是为了Reshape节点的形状，只有输出，没有输入
                output_name = node.output[0]
                secure_model.output[output_name] = ssf.ShareFloat(attributes["value"], party.party, party, device)

            elif node.op_type == "Unsqueeze":
                dim = secure_model.output[node.input[1]]

                layer = layers.SecUnsqueeze(dim)
                layer.set_input_and_output(node.input[0], node.output[0])
                secure_model.add(layer)


            elif node.op_type == "Reshape":
                layer = layers.SecReshape()
                layer.set_input_and_output((node.input[0], node.input[1]), node.output[0])
                secure_model.add(layer)

            elif node.op_type == "Pad":
                mode = attributes["mode"]

                layer = layers.SecPad(mode)
                layer.set_input_and_output((node.input[0], node.input[1]), node.output[0])
                secure_model.add(layer)

            elif node.op_type == "Concat":
                axis = attributes["axis"]

                layer = layers.SecConcat(axis)
                layer.set_input_and_output((node.input[0], node.input[1]), node.output[0])
                secure_model.add(layer)

            # elif node.op_type == "BatchNormalization":
            #     # # Y = (X - running_mean) / sqrt(running_var + eps) * gamma + beta # #
            #     epsilon = torch.tensor(attributes["epsilon"])
            #     print("epsilon", epsilon)
            #
            #     if node.input[1] in parameters:
            #         gamma0, gamma1 = ssf.share_float(ssf.encode(parameters[node.input[1]]))  # 乘法，需要分享
            #         print("weight", parameters[node.input[1]])
            #     else:
            #         gamma0, gamma1 = ssf.share_float(ssf.encode(secure_model.output[node.input[1]].value))
            #         print("weight", secure_model.output[node.input[1]].value)
            #
            #     if node.input[2] in parameters:
            #         beta0, beta1 = ssf.share_float(ssf.encode(parameters[node.input[2]]))  # 加法，需要分享
            #         print("bias", parameters[node.input[2]])
            #     else:
            #         beta0, beta1 = ssf.share_float(ssf.encode(secure_model.output[node.input[2]].value))  # 加法，需要分享
            #         print("bias", secure_model.output[node.input[2]].value)
            #
            #     if node.input[3] in parameters:
            #         running_mean0, running_mean1 = ssf.share_float(ssf.encode(parameters[node.input[3]]))  # 减法，需要分享
            #         print("running_mean", parameters[node.input[3]])
            #     else:
            #         running_mean0, running_mean1 = ssf.share_float(ssf.encode(secure_model.output[node.input[3]].value))  # 减法，需要分享
            #         print("running_mean", secure_model.output[node.input[3]].value)
            #
            #     if node.input[4] in parameters:
            #         running_var = parameters[node.input[4]]
            #     else:
            #         running_var = secure_model.output[node.input[4]].value
            #     print("running_var", running_var)
            #
            #
            #
            #     party.send_torch_array(gamma0)
            #     party.send_torch_array(beta0)
            #     party.send_torch_array(running_mean0)
            #     party.send_torch_array(running_var)
            #
            #     layer = layers.SecBN2d(ssf.ShareFloat(gamma1, party.party, party),
            #                            ssf.ShareFloat(beta1, party.party, party),
            #                            ssf.ShareFloat(running_mean1, party.party, party), running_var, epsilon)
            #     layer.set_input_and_output(node.input[0], node.output[0])
            #     secure_model.add(layer)
            else:
                pass

    if party.party == 0:
        for node in onnx_model.graph.node:
            # get attributes and node type:
            attributes = {attr.name: _get_attribute_value(attr) for attr in node.attribute}

            if node.op_type == "Identity":
                value = party.receive_torch_array(device)
                secure_model.output[node.output[0]] = ssf.ShareFloat(value, party.party, party, device)

            if node.op_type == "Conv":
                pads = attributes["pads"]
                strides = attributes["strides"]

                w0 = party.receive_torch_array(device)
                if len(node.input) == 2:
                    b0 = None
                else:
                    b0 = party.receive_torch_array(device)

                layer = layers.SecConv2d(weight=w0, stride=strides, padding=pads, bias=b0, device=device)
                layer.set_input_and_output(node.input[0], node.output[0])
                secure_model.add(layer)

            elif node.op_type == "Relu":
                layer = layers.SecReLu()
                layer.set_input_and_output(node.input[0], node.output[0])
                secure_model.add(layer)

            elif node.op_type == "MaxPool":
                pads = attributes["pads"]
                strides = attributes["strides"]
                kernel_shape = attributes["kernel_shape"]

                layer = layers.SecMaxPool2D(kernel_shape, strides, pads, device)
                layer.set_input_and_output(node.input[0], node.output[0])
                secure_model.add(layer)

            elif node.op_type == "AveragePool":

                pads = attributes["pads"]
                strides = attributes["strides"]
                kernel_shape = attributes["kernel_shape"]

                layer = layers.SecAvgPool2D(kernel_shape, strides, pads)
                layer.set_input_and_output(node.input[0], node.output[0])
                secure_model.add(layer)

            elif node.op_type == "Gemm":
                w0 = party.receive_torch_array(device)
                b0 = party.receive_torch_array(device)

                layer = layers.SecGemm(w0, b0, device)
                layer.set_input_and_output(node.input[0], node.output[0])
                secure_model.add(layer)

            elif node.op_type == "MatMul":
                layer = layers.SecMatMul()
                layer.set_input_and_output((node.input[0], node.input[1]), node.output[0])
                secure_model.add(layer)

            elif node.op_type == "Add":
                # 如果加法的一方是linear层的bias，需要特殊处理
                if node.input[0].find("bias") != -1:
                    b0 = party.receive_torch_array(device)
                    secure_model.output[node.input[0]] = ssf.ShareFloat(b0, party.party, party, device)

                layer = layers.SecADD()
                layer.set_input_and_output((node.input[0], node.input[1]), node.output[0])
                secure_model.add(layer)

            elif node.op_type == "Transpose":
                w0 = party.receive_torch_array(device)
                layer = layers.SecTranspose(ssf.ShareFloat(w0, party.party, party, device))
                layer.set_input_and_output(node.input[0], node.output[0])
                secure_model.add(layer)

            elif node.op_type == "Constant":  # 这个节点的value 是为了Reshape节点的形状，只有输出，没有输入
                secure_model.output[node.output[0]] = ssf.ShareFloat(attributes["value"], party.party, party, device)

            elif node.op_type == "Unsqueeze":
                dim = secure_model.output[node.input[1]]
                layer = layers.SecUnsqueeze(dim)
                layer.set_input_and_output(node.input[0], node.output[0])
                secure_model.add(layer)

            elif node.op_type == "Reshape":
                layer = layers.SecReshape()
                layer.set_input_and_output((node.input[0], node.input[1]), node.output[0])
                secure_model.add(layer)

            elif node.op_type == "Pad":
                mode = attributes["mode"]

                layer = layers.SecPad(mode)
                layer.set_input_and_output((node.input[0], node.input[1]), node.output[0])
                secure_model.add(layer)

            elif node.op_type == "Concat":
                axis = attributes["axis"]

                layer = layers.SecConcat(axis)
                layer.set_input_and_output((node.input[0], node.input[1]), node.output[0])
                secure_model.add(layer)

            # elif node.op_type == "BatchNormalization":
            #     # # Y = (X - running_mean) / sqrt(running_var + eps) * gamma + beta # #
            #     epsilon = attributes["epsilon"]
            #
            #     gamma0 = party.receive_torch_array(device)
            #     beta0 = party.receive_torch_array(device)
            #     running_mean0 = party.receive_torch_array(device)
            #     running_var = party.receive_torch_array(device)
            #
            #     layer = layers.SecBN2d(ssf.ShareFloat(gamma0, party.party, party),
            #                            ssf.ShareFloat(beta0, party.party, party),
            #                            ssf.ShareFloat(running_mean0, party.party, party), running_var, epsilon)
            #     layer.set_input_and_output(node.input[0], node.output[0])
            #     secure_model.add(layer)

            else:
                pass
    return secure_model


def _load_onnx_model(onnx_string_or_file):
    """
    Loads ONNX model from file or string.
    """
    if hasattr(onnx_string_or_file, "seek"):
        onnx_string_or_file.seek(0)
        return onnx.load(onnx_string_or_file)
    return onnx.load_model_from_string(onnx_string_or_file)


def _get_input_output_names(onnx_model):
    """
    Return input and output names of the ONNX graph.
    """
    input_names = [input.name for input in onnx_model.graph.input]
    output_names = [output.name for output in onnx_model.graph.output]
    assert len(input_names) >= 1, "number of inputs should be at least 1"
    assert len(output_names) == 1, "number of outputs should be 1"
    return input_names, output_names


def _get_attribute_value(attr):
    """
    Retrieves value from an ONNX attribute.
    """
    if attr.HasField("f"):  # floating-point attribute
        return attr.f
    elif attr.HasField("i"):  # integer attribute
        return attr.i
    elif attr.HasField("s"):  # string attribute
        return attr.s
    elif attr.HasField("t"):  # tensor attribute
        return torch.from_numpy(numpy_helper.to_array(attr.t))
    elif len(attr.ints) > 0:
        return list(attr.ints)
    elif len(attr.floats) > 0:
        return list(attr.floats)
    raise ValueError("Unknown attribute type for attribute %s." % attr.name)
