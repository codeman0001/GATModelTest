import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        self.linear1 = nn.Linear(in_dim, out_dim, bias=False)
        self.linear2 = nn.Linear(2*out_dim, 1, bias=False)

    def edge_attention(self, edges):
        Z = torch.cat((edges.src['h'],edges.dst['h']),1)
        a = self.linear2(Z)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'h': edges.src['h'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], 1)
        h = torch.sum(alpha*nodes.mailbox['h'], 1)
        return {'h': h}

    def forward(self, h):
        z = self.linear1(h)
        self.g.ndata['h'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')