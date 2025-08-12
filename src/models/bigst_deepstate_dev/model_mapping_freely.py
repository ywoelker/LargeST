import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .linear_conv import *
from torch.autograd import Variable
import pdb

class Model(nn.Module):
    def __init__(self, seq_num, in_dim, out_dim, hid_dim, num_nodes, tau, random_feature_dim, node_emb_dim, time_emb_dim, \
                 use_residual, use_bn, use_spatial, use_long, dropout, time_of_day_size, day_of_week_size, supports=None, edge_indices=None):
        super(Model, self).__init__()

        self.tau = tau
        self.layer_num = 3
        self.in_dim = in_dim
        self.random_feature_dim = random_feature_dim
        
        self.use_residual = use_residual
        self.use_bn = use_bn
        self.use_spatial = use_spatial
        self.use_long = use_long
        
        self.dropout = dropout
        self.activation = nn.ReLU()
        self.supports = supports
        
        self.time_num = time_of_day_size
        self.week_num = day_of_week_size
        

        self.num_dsn_nodes = num_nodes
        self.mapping_from_sensors_to_deep_state_nodes = torch.eye(self.num_dsn_nodes, num_nodes, dtype=torch.float32)

        # node embedding layer
        self.node_emb_layer = nn.Parameter(torch.empty(num_nodes, node_emb_dim // 2), requires_grad=True)
        nn.init.xavier_uniform_(self.node_emb_layer)
        self.deep_node_emb_layer = nn.Parameter(torch.empty(self.num_dsn_nodes, node_emb_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.deep_node_emb_layer)
        # normalize the node embedding layer to length = 1
        # self.deep_node_emb_layer = F.normalize(self.deep_node_emb_layer, p=2, dim=1)
        
        # time embedding layer
        self.time_emb_layer = nn.Parameter(torch.empty(self.time_num, time_emb_dim))
        nn.init.xavier_uniform_(self.time_emb_layer)
        self.week_emb_layer = nn.Parameter(torch.empty(self.week_num, time_emb_dim))
        nn.init.xavier_uniform_(self.week_emb_layer)

        # embedding layer
        # self.input_emb_layer_context = nn.Sequential(

        #     nn.Conv2d(in_dim - 1 ,hid_dim, kernel_size=(1, 1), bias=True),
        #     nn.ReLU(),
        #     nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, 1), bias=True),
        #     nn.GLU(dim = 1)

        # )
        
        self.input_emb_layer_context = nn.Conv2d(in_dim - 1 , hid_dim, kernel_size=(1, 1), bias=True)
        

        self.input_emb_layer = nn.Conv2d(seq_num * 3, hid_dim, kernel_size=(1, 1), bias=True)
        
        self.W_1 = nn.Conv2d(node_emb_dim+time_emb_dim*2, hid_dim, kernel_size=(1, 1), bias=True)
        self.W_2 = nn.Conv2d(node_emb_dim+time_emb_dim*2, hid_dim, kernel_size=(1, 1), bias=True)
        self.W_3 = nn.Conv2d(node_emb_dim+time_emb_dim*2, hid_dim, kernel_size=(1, 1), bias=True)
        self.W_4 = nn.Conv2d(node_emb_dim+time_emb_dim*2, hid_dim, kernel_size=(1, 1), bias=True)
        
        self.linear_conv = nn.ModuleList()
        self.bn = nn.ModuleList()

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)
        
        for _ in range(self.layer_num + 2):
            self.linear_conv.append(linearized_conv(hid_dim*4, hid_dim*4, self.dropout, self.tau, self.random_feature_dim))
            self.bn.append(nn.LayerNorm(hid_dim*4))
        
        if self.use_long:
            self.regression_layer = nn.Conv2d(hid_dim*4*2+hid_dim+seq_num, out_dim, kernel_size=(1, 1), bias=True)
        else:
            self.regression_layer = nn.Conv2d(hid_dim*4 * 2, out_dim, kernel_size=(1, 1), bias=True)






    def forward(self, x, feat=None):
        
        # x: (B, N, T, D)
        B, N, T, D = x.size()

        original_x = x[..., :3] # shape (B, N, T, 3)
        contextual = x[...,0 ,1:] # shape (B, N, D-1)
        
        time_emb = self.time_emb_layer[(x[:, :, -1, 1]*self.time_num).type(torch.LongTensor)]
        week_emb = self.week_emb_layer[(x[:, :, -1, 2]).type(torch.LongTensor)]
        
        # input embedding
        x = original_x.contiguous().view(B, N, -1).transpose(1, 2).unsqueeze(-1) # (B, D*T, N, 1)
        input_emb = self.input_emb_layer(x)

        # node embeddings
        x = contextual.contiguous().view(B, N, -1).transpose(1, 2).unsqueeze(-1) # (B, D-1, N, 1)
        node_emb_dyn = self.input_emb_layer_context(x) # (B, dim/2, N, 1)
        # node_emb_static = self.node_emb_layer.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1) # (B, dim/2, N, 1)#
        
        node_emb = node_emb_dyn # torch.cat([node_emb_static, node_emb_dyn], dim=1) # (B, dim*2, N, 1)


        # time embeddings
        time_emb = time_emb.transpose(1, 2).unsqueeze(-1) # (B, dim, N, 1)
        week_emb = week_emb.transpose(1, 2).unsqueeze(-1) # (B, dim, N, 1)


        
        x_g = torch.cat([node_emb, time_emb, week_emb], dim=1) # (B, dim*4, N, 1)
        x = torch.cat([input_emb, node_emb, time_emb, week_emb], dim=1) # (B, dim*4, N, 1)

        # linearized spatial convolution
        x_pool = [x] # (B, dim*4, N, 1)
        deep_node_embed = self.deep_node_emb_layer.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1) # (B, dim, 42, 1)
        
        deep_time_ebm = time_emb[:, :, 0:1, :].expand(-1, -1, self.num_dsn_nodes, -1)
        deep_week_emb = week_emb[:, :, 0:1, :].expand(-1, -1, self.num_dsn_nodes, -1) # (B, dim, 42, 1)

        deep_x_g = torch.cat([deep_node_embed, deep_time_ebm, deep_week_emb], dim=1) # (B, dim*3, 42, 1)
        
        deep_node_query_vec = self.W_1(deep_x_g) # (B, dim, N, 1)
        obs_node_key_vec = self.W_2(x_g) # (B, dim, N, 1)
        deep_node_query_vec = deep_node_query_vec.permute(0, 2, 3, 1) # (B, N, 1, dim)
        obs_node_key_vec = obs_node_key_vec.permute(0, 2, 3, 1) # (B, N, 1, dim)


        # Mapping to Deep State Nodes 

        x, deep_node_query_vec_prime, obs_node_key_vec_prime = self.linear_conv[0](x, deep_node_query_vec, obs_node_key_vec) # (B, dim * 4, N, 1)



        deep_node_key_vec = self.W_3(deep_x_g) # (B, dim, 42, 1)
        deep_node_key_vec = deep_node_key_vec.permute(0, 2, 3, 1) # (B, 42, 1, dim)


        for i in range(1, self.layer_num + 1):
            if self.use_residual:
                residual = x

            x, _, _ = self.linear_conv[i](x, deep_node_key_vec, deep_node_query_vec)
            
            if self.use_residual:
                x = x+residual 
                
            if self.use_bn:
                x = x.permute(0, 2, 3, 1) # (B, N, 1, dim*4)
                x = self.bn[i](x)
                x = x.permute(0, 3, 1, 2)

        x = self.activation(x) # (B, dim * 4 * 2, N, 1)

        obs_node_query_vec = self.W_4(x_g) # (B, dim, N, 1)
        obs_node_query_vec = obs_node_query_vec.permute(0, 2, 3, 1)
            
        x, obs_node_query_vec_prime, deep_node_key_vec_prime = self.linear_conv[-1](x, obs_node_query_vec, deep_node_key_vec) # (B, dim * 4 * 2, N, 1)

        x_pool.append(x)
        x = torch.cat(x_pool, dim=1) # (B, dim*4, N, 1)
        
        x = self.activation(x) # (B, dim*4, N, 1)
        
        if self.use_long:
            feat = feat.permute(0, 2, 1).unsqueeze(-1) # (B, F, N, 1)
            x = torch.cat([x, feat], dim=1)
            x = self.regression_layer(x) # (B, N, T)
            x = x.squeeze(-1).permute(0, 2, 1)
        else:
            x = self.regression_layer(x) # (B, N, T)
            x = x.squeeze(-1).permute(0, 2, 1)

        return {"prediction": x.transpose(1,2).unsqueeze(-1)
              , "node_vec1": obs_node_query_vec_prime
              , "node_vec2": obs_node_key_vec_prime
              , "supports": self.supports
              , 'use_spatial': self.use_spatial
              , 'deep_node_vec': self.deep_node_emb_layer}