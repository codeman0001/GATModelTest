from MultiHeadGAT import MultiHeadGAT
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import networkx as nx
import torch.optim as optim
import time


class GATModel(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, head_num):
        super(GATModel, self).__init__()
        self.gat1 = MultiHeadGAT(g, in_dim, hidden_dim, head_num)
        self.gat2 = MultiHeadGAT(g, hidden_dim*head_num, out_dim, 1)

    def forward(self, h):
        h = self.gat1(h)
        h = F.elu(h)
        h = self.gat2(h)
        return h


def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    #采样训练数据，mask为布尔值，True代表训练数据
    mask = torch.BoolTensor(data.train_mask)
    g = data.graph
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, mask


if __name__ == '__main__':
    g, features, labels, mask = load_cora_data()
    # print(features)
    net = GATModel(g, features.size()[1], 8, 7, 2)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    #训练模型
    for epoch in range(800):
        t0 = time.time()
        logits = net(features)
        # print(logits)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        run_time = time.time() - t0
        print('Epoch{:d}: loss:{:.4f} | run_time:{:.4f}'.format(epoch, loss.item(), run_time))
    # print(net(features))

    #验证集
    data = citegrh.load_cora()
    val_mask = torch.BoolTensor(data.val_mask)
    logits = net(features)
    val_Y = labels[val_mask].view(-1, 1).float()
    proY = torch.softmax(logits[val_mask], 1)
    argmax_Y = torch.max(proY, 1)[1].view(-1, 1).float()

    print('Accuracy of argmax predictions on the val set: {:4f}%'.format((val_Y == argmax_Y).sum().item()/val_Y.shape[0]*100))