import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader


class SignedGCNConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SignedGCNConv, self).__init__()
        self.conv1_pos = nn.Linear(in_channels, hidden_channels)
        self.conv1_neg = nn.Linear(in_channels, hidden_channels)
        self.conv2_pos = nn.Linear(hidden_channels, out_channels)
        self.conv2_neg = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, pos_edge_index, neg_edge_index):
        x_pos = F.relu(self.conv1_pos(x))
        x_neg = F.relu(self.conv1_neg(x))
        x = x_pos - x_neg

        x_pos = self.conv2_pos(x)
        x_neg = self.conv2_neg(x)
        x = x_pos - x_neg
        return F.log_softmax(x, dim=1)

num_samples = 10
num_nodes = 62
num_features = 5
hidden_channels = 16
out_channels = 4  # 假设二分类任务

# 创建随机的正负边
def generate_edges(num_nodes):
    pos_edge_index = torch.randint(0, num_nodes, (2, num_nodes))
    neg_edge_index = torch.randint(0, num_nodes, (2, num_nodes))
    return pos_edge_index, neg_edge_index

# 创建 DataLoader
data_list = []
for i in range(num_samples):
    x = torch.randn(num_nodes, num_features)  # 节点特征
    pos_edge_index, neg_edge_index = generate_edges(num_nodes)  # 随机生成正负边
    y = torch.randint(0, out_channels, (num_nodes,))  # 随机标签（假设每个节点都有一个标签）

    # 创建图数据
    data = Data(x=x, pos_edge_index=pos_edge_index, neg_edge_index=neg_edge_index, y=y)
    data_list.append(data)

loader = DataLoader(data_list, batch_size=2, shuffle=True)

# 训练模型
def train():
    model.train()
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.pos_edge_index, data.neg_edge_index)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        print("Loss:", loss.item())


