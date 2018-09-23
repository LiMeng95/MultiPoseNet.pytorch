import torch.nn as nn
import torch.nn.init


class BaseModel(nn.Module):
    """
    BaseModel for training and testing
    save all information you need in `saved_for_loss`
    save all information you want to log in `saved_for_log`
    """
    feat_stride = None

    # saved_for_loss = []
    # saved_for_log = OrderedDict()

    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *inputs):
        """
        Forward
        :param inputs: a set of Variable
        :return: output, a Variable or a tuple of Variable
        :return: saved_for_loss, a list of Variable
        """
        raise NotImplementedError

    @staticmethod
    def build_loss(saved_for_loss, *gts):
        """
        build loss Variable after forward
        save all information you need in `saved_for_loss`
        save all information you want to log in `saved_for_log`, an OrderedDict()
        
        :return: loss, a Variable
        :return: saved_for_log, an OrderedDict
        """
        raise NotImplementedError

    def init_weight(self, std=0.1):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight.data, std=std)
                if m.bias is not None:
                    nn.init.constant(m.bias.data, 0.1)
            elif isinstance(m, nn.Linear):
                nn.init.normal(m.weight.data, std=std)
                if m.bias is not None:
                    nn.init.constant(m.bias.data, 0.1)

    @property
    def is_cuda(self):
        p = next(self.parameters())
        return p.is_cuda
