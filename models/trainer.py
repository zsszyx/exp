import torch
from copy import deepcopy
from utils.trick_util import center_loss, CrossCameraTripletLoss, distillation_loss, EMA, jensen_shannon_divergence
from utils.meters import AverageMeter
import numpy as np
from torch.nn import functional as F
from torch import nn, autograd
from IPython import embed
import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer(nn.Module):
    def __init__(self, model: nn.Module, iters, temp=0.1, margin=0.2, temperature=1, momentum=0.1):
        super().__init__()
        self.model = model.to(device)
        self.iters = iters
        # 拷贝模型参数
        self.temp = temp
        self.trpilet = CrossCameraTripletLoss(margin=margin)
        self.temperature = temperature
        self.momentum = momentum

    def classifier(self, features: torch.Tensor, labels):
        outputs = cm(features, labels, self.centrals, self.momentum)
        outputs /= self.temp
        inner_product = torch.exp(outputs)
        ups = torch.gather(inner_product, 1, labels.view(-1, 1))
        ups = ups.view(-1)
        down = torch.sum(inner_product, dim=1, keepdim=False)
        return ups / down

    def start(self, data_loader, epoch, optim, cam_loader=None):
        self.model.train()
        losses = AverageMeter()
        for i in range(self.iters):
            # extract
            data, _, label, old_label, cid, _ = data_loader.next()
            data = data.to(device)
            label = label.to(device)
            # cams = cid.to(device)
            new_f = self._forward(data)
            new_f = F.normalize(new_f, dim=1).to(device)

            # loss
            new_p = self.classifier(new_f, label)
            contrast_loss = -torch.mean(torch.log(new_p))

            # cam domain
            cam_data1, cam_data2 = cam_loader.next()
            cam_data1 = cam_data1.to(device)
            cam_data2 = cam_data2.to(device)
            cam_data1 = self._forward(cam_data1)
            cam_data2 = self._forward(cam_data2)
            coral_loss = coral(cam_data1, cam_data2)

            # sum
            loss = contrast_loss + coral_loss

            # optim
            optim.zero_grad()
            loss.backward()
            # self.update(new_f, label, cams)
            optim.step()

            # record
            record = np.array([loss.item(),
                               contrast_loss.item(),
                               coral_loss.item()
                               ])
            losses.update(record)
            if (i + 1) % 20 == 0:
                print('Epoch: [{}][{}/{}]\n'
                      'loss {:.3f} ({:.3f})    | '
                      'contrast {:.3f} ({:.3f}) '
                      'coral {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              losses.val[0], losses.avg[0],
                              losses.val[1], losses.avg[1],
                              losses.val[2], losses.avg[2],
                              ))

    def _forward(self, inputs):
        return self.model(inputs)


def coral(source, target):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)
    # source covariance
    tmp_s = torch.ones((1, ns)).to(device) @ source
    cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

    # target covariance
    tmp_t = torch.ones((1, nt)).to(device) @ target
    ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

    # 计算源域和目标域之间的 Frobenius 范数
    loss = torch.norm(cs - ct, p='fro') ** 2 / (4 * d * d)
    return loss


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))
