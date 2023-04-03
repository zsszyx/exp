from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from torch import nn
from torch.nn import init
from abc import ABC
from torch.nn import functional as F
import torch
from .cotlayer import CoTLayer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class my_resnet(nn.Module):
    def __init__(self, model):
        super().__init__()
        out_planes = model.fc.in_features
        feat_bn = nn.BatchNorm1d(out_planes)
        feat_bn.bias.requires_grad_(False)
        init.constant_(feat_bn.weight, 1)
        init.constant_(feat_bn.bias, 0)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.gem = GeneralizedMeanPoolingP()
        self.flatten = FlattenLayer()
        self.featbn = feat_bn
        self.norm = NormlizeLayer()
        self.pt1 = CotBridge(256)
        # self.pt2 = CotBridge(512)
        # self.pt0 = CotBridge(64, expand=8, downsample=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # skip0 = self.pt0(x)
        x = self.layer1(x)
        x0 = x

        skip = self.pt1(x0)
        x0 = self.layer2(x0)
        x1 = x0 + skip

        # skip1 = self.pt2(x1)
        x1 = self.layer3(x1)
        x2 = x1

        outs = self.layer4(x2)

        outs = self.gem(outs)
        outs = self.flatten(outs)
        outs = self.featbn(outs)
        outs = self.norm(outs)
        return outs


def create(load=False):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.layer4[0].conv2.stride = (1, 1)
    model.layer4[0].downsample[0].stride = (1, 1)
    model = my_resnet(model)
    if load:
        new = torch.load('models/model_best.pth.tar', map_location=torch.device('cpu'))['state_dict']
        old = model.state_dict()
        for key in old.keys():
            old[key] = new['module.'+key]
        model.load_state_dict(old)
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


class PartConv(nn.Module):
    def __init__(self, in_channels: int, stride=2, kernel_size=2):
        super().__init__()
        pad = 0
        if stride == 1:
            pad = 1
        self.up_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                 stride=stride, bias=False, padding=pad)
        self.down_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                   stride=stride, bias=False, padding=pad)
        self.global_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                     stride=stride, bias=False, padding=pad)
        init.zeros_(self.up_conv.weight)
        init.zeros_(self.down_conv.weight)
        init.zeros_(self.global_conv.weight)

    def forward(self, x):
        assert x.shape[2] % 2 == 0
        cut = int(x.shape[2] / 2)
        up = x[:, :, :cut, :]
        down = x[:, :, cut:, :]

        up = self.up_conv(up)

        down = self.down_conv(down)

        glob = self.global_conv(x)

        component = torch.cat((up, down), dim=2)
        outputs = torch.cat((glob, component), dim=1)
        indices = torch.randperm(outputs.shape[1])
        outputs = outputs[:, indices, :, :]
        outputs = F.dropout(outputs, p=0.5, training=self.training)
        return outputs


class CotBridge(nn.Module):
    def __init__(self, in_channels: int, kernel_size=3, downsample=True, expand=2):
        super().__init__()
        assert kernel_size % 2 == 1
        self.cot = CoTLayer(in_channels, kernel_size)
        if downsample:
            self.ds = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * expand, kernel_size=2, stride=2,
                                bias=False)
        else:
            self.ds = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * expand, kernel_size=3, stride=1,
                                bias=False, padding=1)
        init.zeros_(self.ds.weight)

    def forward(self, x):
        x = self.cot(x)
        x = self.ds(x)
        indices = torch.randperm(x.shape[1])
        outputs = x[:, indices, :, :]
        outputs = F.dropout(outputs, p=0.5, training=self.training)
        return outputs
# a = create()
# a.training=True
# b = torch.rand([32,3,64,32])
# c = a(b).sum()
# c.backward()
