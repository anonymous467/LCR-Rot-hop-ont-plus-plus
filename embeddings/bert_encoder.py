# https://github.com/autoliuweijie/K-BERT
# -*- encoding:utf-8 -*-
import torch.nn as nn
from layer_norm import LayerNorm
from position_ffn import PositionwiseFeedForward
from multi_headed_attn import MultiHeadedAttention
from transformer import TransformerLayer


class BertEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """
    def __init__(self, args, model):
        super(BertEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.transformer = nn.ModuleList([
            TransformerLayer(args, model.base_model.encoder.layer._modules.get(key)) for key in model.base_model.encoder.layer._modules
        ])
        
        
    def forward(self, emb, seg, vm=None):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
            vm: [batch_size x seq_length x seq_length]

        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        seq_length = emb.size(1)
        # Generate mask according to segment indicators.
        # mask: [batch_size x 1 x seq_length x seq_length]
        hidden_layers = []
        hidden = emb
        for i in range(self.layers_num):
            hidden = self.transformer[i](hidden, vm)
            hidden_layers.append(hidden)

        # take the average of the last 4 layers
        hidden = (hidden_layers[11] + hidden_layers[10] + hidden_layers[9]+ hidden_layers[8])/4
        return hidden
