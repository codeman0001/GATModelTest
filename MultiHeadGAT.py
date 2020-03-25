from GATLayer import GATLayer
import torch
import torch.nn as nn


class MultiHeadGAT(nn.Module):
    def __init__(self, g, in_dim, out_dim, head_num, merge='cat'):
        super(MultiHeadGAT, self).__init__()
        self.multiHeadGAT = nn.ModuleList([GATLayer(g, in_dim, out_dim) for k in range(head_num)])
        self.merge = merge

    def forward(self, h):
        head_out = [multiHead(h) for multiHead in self.multiHeadGAT]
        if self.merge == 'cat':
            return torch.cat(head_out, 1)
        else:
            return torch.mean(torch.stack(head_out))