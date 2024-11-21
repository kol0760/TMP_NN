import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch import nn
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def re_encoded_edge(tensor):
    unique_values = torch.unique(tensor)
    sorted_values = unique_values.sort().values
    value_to_new_value = {value.item(): index for index, value in enumerate(sorted_values)}
    re_encoded_tensor = torch.tensor([[value_to_new_value[value.item()] for value in row] for row in tensor])
    return re_encoded_tensor


def encode_features(x_feature, num_bins=5):
    feature_data = x_feature[:, 0:22] # The first 22 data are continuous variables

    def quantile_bins(data, num_bins):
        return np.quantile(data, np.linspace(0, 1, num_bins + 1))

    encoded_features = []
    for i in range(feature_data.shape[0]):
        bins = quantile_bins(feature_data[i], num_bins)
        binned_data = np.digitize(feature_data[i], bins, right=True)
        binned_data = np.clip(binned_data, 0, num_bins - 1)
        one_hot_encoded = np.eye(num_bins)[binned_data]
        encoded_features.append(one_hot_encoded.flatten())

    encoded_features = np.array(encoded_features)
    Total_features = np.concatenate((encoded_features, x_feature[:, 23:]), axis=1)
    return Total_features

def get_sub_G(original_data, index_list):
    edge_index = original_data.edge_index
    edge_attr = original_data.edge_attr
    subset = torch.tensor(index_list)
    new_edge_index, new_edge_attr = subgraph(subset, edge_index, edge_attr)
    new_edge_index= re_encoded_edge(new_edge_index)
    subgraph_data = Data(
        x=original_data.x[index_list],
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        y=original_data.y,
        pos=original_data.pos[index_list] if original_data.pos is not None else None,
        global_features=original_data.global_features,
        x3d=original_data.x3d[index_list] if original_data.x3d is not None else None,
        x2d=original_data.x2d[index_list] if original_data.x2d is not None else None
    )
    return subgraph_data

from torch_geometric.data import Data

def get_sub_G_list(original_data):
    fragments = list(original_data.fragment)
    fragments[0].extend([0])
    fragments[0].sort()
    fragments[1].extend([5])
    fragments[1].sort()
    fragments[2].extend([0,1,4,5])
    fragments[2].sort()
    index_1 = fragments[0]
    index_2 = fragments[1]
    index_3 = fragments[2]
    subgraph_1 = get_sub_G(original_data,index_1)
    subgraph_2 = get_sub_G(original_data,index_2)
    subgraph_3 = get_sub_G(original_data,index_3)
    return subgraph_1,subgraph_2,subgraph_3,original_data.y


class GCN_PyG(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN_PyG, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, hidden_features//2)
        self.conv3 = GCNConv(hidden_features//2, hidden_features//4)
        self.dropout1 = nn.Dropout(p=0.1)  
        self.dropout2 = nn.Dropout(p=0.2)  
    def forward(self, subgraph_1, subgraph_2, subgraph_3):
        x1,edge_index1 = subgraph_1.x,subgraph_1.edge_index
        x2,edge_index2 = subgraph_2.x,subgraph_2.edge_index
        x3,edge_index3 = subgraph_3.x,subgraph_3.edge_index
        x1 = F.relu(self.conv1(x1, edge_index1))
        x1 = self.dropout1(x1)  
        x1 = F.relu(self.conv2(x1, edge_index1))
        x1 = self.dropout1(x1)  
        x1 = F.relu(self.conv3(x1, edge_index1))

        x2 = F.relu(self.conv1(x2, edge_index2))
        x2 = self.dropout1(x2)  
        x2 = F.relu(self.conv2(x2, edge_index2))
        x2 = self.dropout1(x2)  
        x2 = F.relu(self.conv3(x2, edge_index2))

        x3 = F.relu(self.conv1(x3, edge_index3))
        x3 = self.dropout2(x3)  
        x3 = F.relu(self.conv2(x3, edge_index3))
        x2 = self.dropout2(x2)  
        x3 = F.relu(self.conv3(x3, edge_index3))

        x = torch.sum(torch.cat([x1,x2,x3],axis=0),axis=0)
        # x = torch.sum(torch.stack([x1, x2, x3], dim=0), dim=0)
        x = torch.mean(x, dim=0)
        return x

