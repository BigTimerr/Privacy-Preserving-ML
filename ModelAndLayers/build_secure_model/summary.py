import torch
import torch.nn as nn
from torch.autograd import Variable
import ProtocolOnRing.secret_sharing_fixpoint as ssf
from collections import OrderedDict
import numpy as np

Qring = 10000

def model_process_summary(model):

    def register_hook(module):
        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()

            if hasattr(module, "weight"):
                weight = module.weight
                weight = ssf.encode(weight)
                # print("module.weight", repr(weight))
                summary[m_key]["weight"] = weight
            if hasattr(module, "bias"):
                bias = module.bias
                if bias is None:
                    bias = torch.zeros(size=(1, 1))
                bias = ssf.encode(bias)
                summary[m_key]["bias"] = bias
            if hasattr(module, "kernel_size"):
                summary[m_key]["kernel_size"] = module.kernel_size
            if hasattr(module, "stride"):
                summary[m_key]["stride"] = module.stride
            if hasattr(module,"padding"):
                summary[m_key]["padding"] = module.padding



    # create properties
    summary = OrderedDict()

    # register hook
    model.apply(register_hook)


    return summary
