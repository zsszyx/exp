#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class LoRALayer():
    def __init__(
            self,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Conv2d(nn.Conv2d, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert type(kernel_size) is int
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels * kernel_size, r * kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = True
        self.reset_parameters()

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = False

    def eval(self):
        nn.Conv2d.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            return F.conv2d(
                x,
                self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        return nn.Conv2d.forward(self, x)


def mark_conv_lora_as_trainable(model: nn.Module) -> None:
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            for p_name, p in m.named_parameters():
                if 'lora_' in p_name:
                    p.requires_grad = True
                else:
                    p.requires_grad = False


def mark_conv_all_as_trainable(model: nn.Module) -> None:
    for name, m in model.named_parameters():
        if isinstance(m, nn.Conv2d):
            for p_name, p in m.named_parameters():
                p.requires_grad = True


def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def low_rank_cov_change(module, rank=16):
    for name, layer in module.named_modules():
        if isinstance(layer, nn.Conv2d):
            out_channel = layer.weight.shape[0]
            in_channel = layer.weight.shape[1]
            core_size = layer.weight.shape[2]
            new_cov = Conv2d(in_channel, out_channel, core_size, rank)
            new_cov.bias = layer.bias
            new_cov.stride = layer.stride
            # new_cov.stride = 1
            new_cov.padding = layer.padding
            new_cov.dilation = layer.dilation
            new_cov.groups = layer.groups
            new_dict = new_cov.state_dict()
            old_dict = layer.state_dict()
            for i in old_dict.keys():
                if i in new_dict.keys():
                    new_dict[i] = old_dict[i]
            new_cov.load_state_dict(new_dict)
            _set_module(module, name, new_cov)
    mark_conv_all_as_trainable(module)
    return module

# a = create()
# # for n, p in a.named_parameters():
# #     print(n, p.shape)
# m = low_rank_cov_change(a)
# # for n, p in m.named_parameters():
# #     print(n, p.shape)
# sample = torch.rand([32,3,256,128])
# x = m(sample)
# print(x.shape)