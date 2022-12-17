from ModelAndLayers.layers.layers_of_fixpoints import SecReLu, SecLinear, SecConv2d, SecAvgPool2D, SecMaxPool2D, Flatten
import ProtocolOnRing.secret_sharing_fixpoint as ssf
from ModelAndLayers.build_secure_model.summary import model_process_summary
from ModelAndLayers.model.modeloflayers import ModelOfLayers

def buildSecureModelProcessOnClient(model, client):
    secure_operators = ModelOfLayers()
    process_orders = model_process_summary(model)
    for key in process_orders.keys():
        if "Linear" == key.split("-")[0]:
            w0, w1 = ssf.share_float(process_orders[key]['weight'])
            b0, b1 = ssf.share_float(process_orders[key]['bias'])
            client.send_torch_array(w1)
            client.send_torch_array(b1)
            op = SecLinear(weight=w0, bias=b0)
            secure_operators.add(op)

        if "Conv2d" == key.split("-")[0]:
            w0, w1 = ssf.share_float(process_orders[key]['weight'])
            b0, b1 = ssf.share_float(process_orders[key]['bias'])
            client.send_torch_array(w1)
            client.send_torch_array(b1)
            op = SecConv2d(weight=w0, stride=process_orders[key]["stride"],padding=process_orders[key]["padding"], bias=b0)
            secure_operators.add(op)
        if "ReLU" == key.split("-")[0]:
            op = SecReLu()
            secure_operators.add(op)

        if "AvgPool2d" == key.split("-")[0]:

            op = SecAvgPool2D(kernel_shape=process_orders[key]["kernel_size"], stride=process_orders[key]["stride"])
            secure_operators.add(op)

        if "MaxPool2d" == key.split("-")[0]:
            op = SecMaxPool2D(kernel_shape=process_orders[key]["kernel_size"],stride=process_orders[key]["stride"])
            secure_operators.add(op)

        if "Flatten" == key.split("-")[0]:
            op = Flatten()
            secure_operators.add(op)

    return secure_operators


def buildSecureModelProcessOnServer(model, server):
    secure_operators = ModelOfLayers()
    process_orders = model_process_summary(model)

    for key in process_orders.keys():

        if "Linear" == key.split("-")[0]:
            w1 = server.receive_torch_array()
            b1 = server.receive_torch_array()

            op = SecLinear(weight=w1, bias=b1)
            secure_operators.add(op)
        if "Conv2d" == key.split("-")[0]:
            w1 = server.receive_torch_array()
            b1 = server.receive_torch_array()
            op = SecConv2d(weight=w1, stride=process_orders[key]["stride"],padding=process_orders[key]["padding"], bias=b1)
            secure_operators.add(op)
        if "ReLU" == key.split("-")[0]:
            op = SecReLu()
            secure_operators.add(op)

        if "AvgPool2d" == key.split("-")[0]:
            op = SecAvgPool2D(kernel_shape=process_orders[key]["kernel_size"], stride=process_orders[key]["stride"])
            secure_operators.add(op)

        if "MaxPool2d" == key.split("-")[0]:
            op = SecMaxPool2D(kernel_shape=process_orders[key]["kernel_size"], stride=process_orders[key]["stride"])
            secure_operators.add(op)

        if "Flatten" == key.split("-")[0]:
            op = Flatten()
            secure_operators.add(op)

    return secure_operators
