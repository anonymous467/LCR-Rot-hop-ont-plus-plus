# https://github.com/autoliuweijie/K-BERT
# -*- encoding:utf-8 -*-
import torch.nn as nn
from act_fun import gelu


class PositionwiseFeedForward(nn.Module):
    """ Feed Forward Layer """
    def __init__(self, hidden_size, feedforward_size, layer):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = layer.intermediate.dense
        self.linear_2 = layer.output.dense
        
    def forward(self, x):
        inter = gelu(self.linear_1(x))
        output = self.linear_2(inter)
        return output
