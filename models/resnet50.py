from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from torch import nn
from torch.nn import init
from abc import ABC
from torch.nn import functional as F
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create(load=False):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    out_planes = model.fc.in_features
    feat_bn = nn.BatchNorm1d(out_planes)
    feat_bn.bias.requires_grad_(False)
    init.constant_(feat_bn.weight, 1)
    init.constant_(feat_bn.bias, 0)
    # model.conv1.stride = (1, 1)
    model.layer4[0].conv2.stride = (1, 1)
    model.layer4[0].downsample[0].stride = (1, 1)
    layers = list(model.children())
    layers.pop()
    model = nn.Sequential(*layers, GeneralizedMeanPoolingP(), FlattenLayer(), feat_bn, NormlizeLayer())
    if load:
        new_state_dict = torch.load('models/model_best.pth.tar', map_location='cpu')['state_dict']
        copy_dict = {}
        for i in new_state_dict.keys():
            copy_dict[i[7:]] = new_state_dict[i]
        old_dict = model.state_dict()
        keys = old_dict.keys()
        for i in copy_dict.keys():
            if i in keys:
                old_dict[i] = copy_dict[i]
        model.load_state_dict(old_dict)
    # model = nn.DataParallel(model)
    return model.to(device)


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class GeneralizedMeanPooling(nn.Module, ABC):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of
    several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size
                     will be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(
            1.0 / self.p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + str(self.p)
            + ", "
            + "output_size="
            + str(self.output_size)
            + ")"
        )


class GeneralizedMeanPoolingP(GeneralizedMeanPooling, ABC):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


class NormlizeLayer(nn.Module):
    def __init__(self):
        super(NormlizeLayer, self).__init__()

    def forward(self, x):
        return F.normalize(x)


# a =create()
# for name, m in a.named_parameters():
#     print(name, m.shape)
