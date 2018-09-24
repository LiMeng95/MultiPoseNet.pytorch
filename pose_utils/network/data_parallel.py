import itertools
import torch
from torch.nn import DataParallel
from torch.autograd import Variable
from torch.nn.parallel._functions import Scatter, Gather


class ScatterList(list):
    pass


class ConstList(list):
    pass


class ListDataParallel(DataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        return pose_scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def gather(self, outputs, output_device):
        return pose_gather(outputs, output_device, dim=self.dim)


def scatter(inputs, target_gpus, dim=0):
    r"""
    Slices variables into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not variables. Does not
    support Tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, Variable):
            return Scatter.apply(target_gpus, None, dim, obj)
        assert not torch.is_tensor(obj), "Tensors not supported in scatter."
        if isinstance(obj, ScatterList):
            assert len(obj) == len(target_gpus)
            return [obj[i] for i in range(len(target_gpus))]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    return scatter_map(inputs)


def pose_scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


def pose_gather(outputs, target_device, dim=0):
    r"""
    Gathers variables from different GPUs on a specified device
      (-1 means the CPU).
    """
    def gather_map(outputs):
        if isinstance(outputs, Variable):
            if target_device == -1:
                return outputs.cpu()
            return outputs.cuda(target_device)

        out = outputs[0]
        if isinstance(out, Variable):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None

        if isinstance(out, str):
            return out
        if isinstance(out, ConstList):
            return out
        if isinstance(out, ScatterList):
            return tuple(map(gather_map, itertools.chain(*outputs)))

        return type(out)(map(gather_map, zip(*outputs)))
    return gather_map(outputs)
