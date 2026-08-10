import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base.model import BaseModel


class ContextOnlyMlp(BaseModel):
    def __init__(self, node_num,input_dim, output_dim, **args):
        super(ContextOnlyMlp, self).__init__(node_num = node_num, input_dim=input_dim, output_dim=output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim-3 + 3*self.seq_len, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, self.horizon * self.output_dim)

    def forward(self, x, label=None):
        
        x_traffic = x[..., :3]
        x_traffic = x_traffic.transpose(1,2).reshape(x_traffic.shape[0], self.node_num, -1)
        x_context = x[:, 0, :, 3:]
        
        x_transformed = torch.cat([x_traffic, x_context], dim=-1)
        
        
        x = F.relu(self.fc1(x_transformed))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.reshape(x.shape[0], self.node_num, self.horizon, self.output_dim).transpose(1,2)
        return x