import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def reparameterize(x, var):
    # 求矩阵x的均值和方差
    mean = x.mean(dim=0, keepdim=False)

    # 将方差加上一个小的常数，避免除以0
    eps = 1e-6
    var = var + eps
    std = torch.exp(0.5 * var)
    eps = torch.randn_like(std)
    # 返回重参数化的结果
    return mean + eps * std


class CrossCameraTripletLoss(torch.nn.Module):
    def __init__(self, margin=0.2):
        super(CrossCameraTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, features, labels):
        # features: [batch_size, feature_size]
        # labels: [batch_size,], num_tasks是每个行人的任务数
        dists = torch.cdist(features, features, p=2)  # 计算特征间的欧几里得距离
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # 获取正样本掩码
        neg_mask = ~pos_mask  # 获取负样本掩码

        # 跨相机triplet选择
        anchors, pos, neg = [], [], []
        for i in range(features.shape[0]):
            anchor = i
            pos_inds = torch.where(pos_mask[i])[0]
            neg_inds = torch.where(neg_mask[i])[0]
            if len(pos_inds) == 0 or len(neg_inds) == 0:
                continue
            pos_dist = dists[i][pos_inds]
            neg_dist = dists[i][neg_inds]

            # 对于每个正样本，找到与它距离最近的负样本
            min_neg_dist, min_neg_index = torch.min(neg_dist), torch.argmin(neg_dist)
            neg_ind = neg_inds[min_neg_index]

            # 对于每个负样本，找到与它距离最远的正样本
            max_pos_dist, max_pos_index = torch.max(pos_dist), torch.argmax(pos_dist)
            pos_ind = pos_inds[max_pos_index]

            anchors.append(features[anchor])
            pos.append(features[pos_ind])
            neg.append(features[neg_ind])

        if len(anchors) == 0:
            return torch.tensor(0.0, requires_grad=True).to(features.device)

        anchors = torch.stack(anchors)
        pos = torch.stack(pos)
        neg = torch.stack(neg)

        # 计算triplet loss
        pos_dists = F.pairwise_distance(anchors, pos, p=2)
        neg_dists = F.pairwise_distance(anchors, neg, p=2)
        loss = torch.mean(F.relu(pos_dists - neg_dists + self.margin))

        return loss


class EMA:
    def __init__(self, delta=0.999):
        self.delta = delta

    def update(self, old, new, model):
        # 在动量更新时修改动量参数
        for name, param in old.items():
            old[name] = param * self.delta + (1. - self.delta) * new[name]

        # 创建一个新的模型，使用拷贝的模型参数
        model.load_state_dict(old)
        for param in model.parameters():
            param.requires_grad = False


def distillation_loss(outputs, teacher_outputs, temperature=0.999):
    """
    计算蒸馏损失

    Args:
    outputs: 学生模型的输出
    teacher_outputs: 教师模型的输出
    temperature: 温度参数，默认为1

    Returns:
    loss: 蒸馏损失
    """
    # 对教师模型的输出进行软化
    # soft_teacher_outputs = F.softmax(teacher_outputs / temperature, dim=0)

    # 计算交叉熵损失
    loss = nn.KLDivLoss(reduction='batchmean')(outputs/temperature, teacher_outputs/temperature)

    # 返回损失
    return loss


def center_loss(inputs, targets, centers):
    batch_size = inputs.size(0)
    num_features = inputs.size(1)
    feat_labels = targets.unsqueeze(1).expand(batch_size, num_features)
    centers_batch = centers.gather(0, feat_labels)
    # 计算中心损失
    loss = torch.sum(torch.pow(inputs - centers_batch, 2)) / 2.0 / batch_size
    return loss


def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    kl_divergence_pm = F.kl_div(p, m, reduction='batchmean')
    kl_divergence_qm = F.kl_div(q, m, reduction='batchmean')
    return torch.sqrt(0.5 * kl_divergence_pm + 0.5 * kl_divergence_qm)


def few_shot(model):
    # 对每一层的参数进行处理
    for name, param in model.named_parameters():
        # 如果是卷积层或线性层的权重参数
        if param.requires_grad and len(param.shape) > 1:
            threshold = np.percentile(torch.abs(param.view(-1)).cpu().numpy(), 10)
            # 将绝对值小于阈值的参数的requires_grad属性设置为True
            param.requires_grad = torch.tensor(param < threshold, dtype=torch.bool)
