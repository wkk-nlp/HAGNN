from typing import Any

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from module.GATStackLayer import MultiHeadLayer
from module.GATLayer import PositionwiseFeedForward, WSGATLayer, SWGATLayer, SAGATLayer, ASGATLayer


class WSWGAT(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, ffn_inner_hidden_size, ffn_drop_out,
                 feat_embed_size, layerType):
        super().__init__()
        self.layerType = layerType
        if layerType == "W2S":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out,
                                        feat_embed_size, layer=WSGATLayer)
        elif layerType == "S2W":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out,
                                        feat_embed_size, layer=SWGATLayer)
        elif layerType == "S2A":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out,
                                        feat_embed_size, layer=SAGATLayer)
        elif layerType == "A2S":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out,
                                        feat_embed_size, layer=ASGATLayer)
        else:
            raise NotImplementedError("GAT Layer has not been implemented!")

        self.ffn = PositionwiseFeedForward(out_dim, ffn_inner_hidden_size, ffn_drop_out)

    def forward(self, g, source, dest):
        if self.layerType == "W2S":
            origin, neighbor = dest, source
        elif self.layerType == "S2W":
            origin, neighbor = dest, source
        elif self.layerType == "S2S":
            assert torch.equal(dest, source)
            origin, neighbor = dest, source
        elif self.layerType == "S2A":
            origin, neighbor = dest, source
        elif self.layerType == "A2S":
            origin, neighbor = dest, source
        else:
            origin, neighbor = None, None

        h = F.elu(self.layer(g, neighbor))
        h = h + origin
        h = self.ffn(h.unsqueeze(0)).squeeze(0)
        return h

