import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


class CoTLayer(nn.Module):
    def __init__(self, dim, kernel_size):
        super(CoTLayer, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size // 2,
                      groups=4 if self.dim % 4 == 0 else 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8 if 8 < self.dim else self.dim
        self.share_planes = share_planes
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv2d(2 * dim, dim // factor, 1, bias=False),
            nn.BatchNorm2d(dim // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.bn = nn.BatchNorm2d(dim)

        self.act = nn.SiLU(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv2d(dim, attn_chs, 1),
            nn.BatchNorm2d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_chs, self.radix * dim, 1)
        )
        self.reset()

    def forward(self, x):
        k = self.key_embed(x)
        qk = torch.cat([x, k], dim=1)
        b, c, qk_hh, qk_ww = qk.size()

        w = self.embed(qk)
        w = w.view(b, 1, -1, self.kernel_size * self.kernel_size, qk_hh, qk_ww)

        x = self.conv1x1(x)
        x = F.unfold(x, self.kernel_size, padding=(self.kernel_size - 1) // 2)
        x = x.view(b, self.share_planes, -1, self.kernel_size * self.kernel_size, qk_hh, qk_ww)
        x = x * w
        x = x.sum(3)
        x = x.view(b, int(c/2), qk_hh, qk_ww)
        x = self.bn(x)
        x = self.act(x)

        B, C, H, W = x.shape
        x = x.view(B, C, 1, H, W)
        k = k.view(B, C, 1, H, W)
        x = torch.cat([x, k], dim=2)

        x_gap = x.sum(dim=2)
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_attn = self.se(x_gap)
        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1))).sum(dim=2)

        return out

    def reset(self):
        init.kaiming_uniform_(self.key_embed[0].weight, nonlinearity='relu')
        init.normal_(self.key_embed[1].weight, mean=1, std=0.02)
        init.constant_(self.key_embed[1].bias, 0)

        init.kaiming_uniform_(self.embed[0].weight, nonlinearity='relu')
        init.normal_(self.embed[1].weight, mean=1, std=0.02)
        init.constant_(self.embed[1].bias, 0)
        init.xavier_uniform_(self.embed[3].weight)
        init.normal_(self.embed[4].weight, mean=1, std=0.02)
        init.constant_(self.embed[4].bias, 0)

        init.xavier_uniform_(self.conv1x1[0].weight)

        init.normal_(self.bn.weight, mean=1, std=0.02)
        init.constant_(self.bn.bias, 0)

        init.kaiming_uniform_(self.se[0].weight, nonlinearity='relu')
        init.normal_(self.se[1].weight, mean=1, std=0.02)
        init.constant_(self.se[1].bias, 0)
        init.xavier_uniform_(self.se[3].weight)


# a = CoTLayer(3, 3)
# b = CoTLayer(64, 3)
# c = torch.rand([10,3,128,64])
# d = torch.rand([10,64,128,64])
# e = a(c)
# f = b(d)
# print(e.shape, f.shape)
# l = f.sum() + e.sum()
# l.backward()