device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, hidden_dim):
        super(CrossAttention, self).__init__()
        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.key_projection = nn.Linear(key_dim, hidden_dim)
        self.value_projection = nn.Linear(value_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, query, key, value):
        query_projected = self.layer_norm(self.query_projection(query))
        key_projected = self.layer_norm(self.key_projection(key))
        value_projected = self.layer_norm(self.value_projection(value))

        # Calculate attention scores
        attention_scores = torch.matmul(query_projected, key_projected.transpose(-2, -1))
        attention_scores = attention_scores / (self.hidden_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply the attention weights to the values
        attention_output = torch.matmul(attention_weights, value_projected)
        return attention_output
    
class WeightedConcat(nn.Module):
    def __init__(self, num_inputs):
        super(WeightedConcat, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, requires_grad=True))

    def forward(self, *inputs):
        weighted_inputs = [w * x for w, x in zip(self.weights, inputs)]
        return torch.cat(weighted_inputs, dim=1)
    
class MultimodalMoleculeNet(nn.Module):
    def __init__(self, in_channels_3d=1, in_channels_2d=1, in_features_rnn=23, 
                 mid_channels=16, out_channels=32, hidden_size_rnn=128, dropout_rate=0.3):
        super(MultimodalMoleculeNet, self).__init__()
        
        # 3D CNN layers
        self.conv3d_1 = nn.Conv3d(in_channels_3d, mid_channels, kernel_size=2)
        self.bn3d_1 = nn.BatchNorm3d(mid_channels)
        self.pool3d = nn.MaxPool3d(2)
        self.conv3d_2 = nn.Conv3d(mid_channels, out_channels, kernel_size=2)
        self.bn3d_2 = nn.BatchNorm3d(out_channels)

        # 2D CNN layers
        self.conv2d_1 = nn.Conv2d(in_channels_2d, mid_channels, kernel_size=2)
        self.bn2d_1 = nn.BatchNorm2d(mid_channels)
        self.pool2d = nn.MaxPool2d(3)
        self.conv2d_2 = nn.Conv2d(mid_channels, out_channels, kernel_size=2)
        self.bn2d_2 = nn.BatchNorm2d(out_channels)
        
        self.attention3d_2d = CrossAttention(query_dim=512, key_dim=200, value_dim=200, hidden_dim=128)
        self.attention1d_2d = CrossAttention(query_dim=23, key_dim=200, value_dim=200, hidden_dim=64)

        # RNN layer
        self.lstm_3d = nn.LSTM(input_size=256, hidden_size=hidden_size_rnn, num_layers=1, batch_first=True)
        self.lstm_2d = nn.LSTM(input_size=32*10, hidden_size=hidden_size_rnn, num_layers=1, batch_first=True)
        self.lstm_rnn = nn.LSTM(input_size=in_features_rnn, hidden_size=hidden_size_rnn,dropout=0.0, num_layers=1, batch_first=True)

        self.lstm_rnn1 = nn.LSTM(input_size=64, hidden_size=hidden_size_rnn,dropout=0.2, num_layers=3, batch_first=True)
        self.lstm_rnn3 = nn.LSTM(input_size=128, hidden_size=hidden_size_rnn,dropout=0.5, num_layers=3, batch_first=True)
        self.weighted_concat = WeightedConcat(num_inputs=5)
        # Fully connected layers

        self.layers = nn.Sequential(

            nn.Linear(5 * hidden_size_rnn, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 64),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x_3d, x_2d, x_rnn):
        batch_size, num_atoms, d, h, w = x_3d.size()
        
        x3d_reshape = x_3d.view(batch_size,num_atoms, -1)
        x2d_reshape = x_2d.view(batch_size, num_atoms, -1)
        x3d_2d = self.attention3d_2d(x3d_reshape, x2d_reshape, x2d_reshape)
        x1d_2d = self.attention1d_2d(x_rnn, x2d_reshape, x2d_reshape)

        
        # 3D CNN branch
        x3d = x_3d.view(batch_size * num_atoms, 1, d, h, w)  # Flatten atomic dimension
        x3d = F.relu(self.bn3d_1(self.conv3d_1(x3d)))
        x3d = self.pool3d(x3d)
        x3d = F.relu(self.bn3d_2(self.conv3d_2(x3d)))

        x3d = x3d.view(batch_size, num_atoms, -1)  # Reshape to (batch_size, num_atoms, features)
        x3d, _ = self.lstm_3d(x3d)
        x3d = x3d[:, -1, :]  # Use the last RNN step output

        # 2D CNN branch
        batch_size, num_atoms, h, w = x_2d.size()
        x2d = x_2d.view(batch_size * num_atoms, 1, h, w)  # Flatten atomic dimension
        x2d = F.relu(self.bn2d_1(self.conv2d_1(x2d)))
        x2d = self.pool2d(x2d)
        x2d = F.relu(self.bn2d_2(self.conv2d_2(x2d)))

        x2d = x2d.view(batch_size, num_atoms, -1)  # Reshape to (batch_size, num_atoms, features)
        x2d, _ = self.lstm_2d(x2d)
        x2d = x2d[:, -1, :]  # Use the last RNN step output

        # RNN branch
        x_rnn, _ = self.lstm_rnn(x_rnn)
        x_rnn = x_rnn[:, -1, :]  # Use the last RNN step output
        
        x3d_2d, _ = self.lstm_rnn3(x3d_2d)
        x3d_2d = x3d_2d[:, -1, :]  # Use the last RNN step output 
        
        x1d_2d, _ = self.lstm_rnn1(x1d_2d)
        x1d_2d = x1d_2d[:, -1, :]  # Use the last RNN step output 
        # print(x1d_2d.shape,x3d_2d.shape)
        # Concatenate outputs from three branches
        x = self.weighted_concat(x3d, x2d, x_rnn, x3d_2d, x1d_2d)
        # x = torch.cat((x3d, x2d, x_rnn,x3d_2d,x1d_2d), dim=1)

        x = self.layers(x)
        
        return x
