import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .linear_conv import *
from torch.autograd import Variable
import pdb
import numpy as np
import numpy as np

def construct_greedy(rows, cols, ones_per_row, ones_per_col):
    """Greedy Havel-Hakimi style construction. Returns exact 0/1 matrix."""
    assert rows * ones_per_row == cols * ones_per_col, "Incompatible totals"
    M = np.zeros((rows, cols), dtype=np.uint8)
    remaining_col = np.full(cols, ones_per_col, dtype=int)
    for r in range(rows):
        # indices sorted by descending remaining capacity
        idxs = np.argsort(-remaining_col)
        # make sure there are enough columns left
        if (remaining_col > 0).sum() < ones_per_row:
            raise RuntimeError(f"Not enough remaining columns for row {r}")
        chosen = idxs[:ones_per_row]
        if (remaining_col[chosen] <= 0).any():
            raise RuntimeError("Attempted to use a column with zero remaining capacity")
        M[r, chosen] = 1
        remaining_col[chosen] -= 1
    return M

def randomize_swaps_edge(M, swaps=100000, rng=None):
    """
    Randomize matrix M by performing 'swaps' successful double-edge swaps.
    A swap picks two current edges (r1,c1) and (r2,c2) and replaces them by
    (r1,c2) and (r2,c1) if those two are currently zero.
    """
    if rng is None:
        rng = np.random.default_rng()
    edges = np.column_stack(np.where(M == 1))  # shape (E, 2)
    E = edges.shape[0]
    if E == 0:
        return M
    attempts = swaps  # number of random attempts (may be tuned)
    for _ in range(attempts):
        i, j = rng.integers(0, E, size=2)
        if i == j:
            continue
        r1, c1 = edges[i]
        r2, c2 = edges[j]
        if r1 == r2 or c1 == c2:
            continue
        # can we swap without creating multi-edge?
        if M[r1, c2] == 0 and M[r2, c1] == 0:
            # perform swap on matrix
            M[r1, c1] = 0
            M[r2, c2] = 0
            M[r1, c2] = 1
            M[r2, c1] = 1
            # update edges list
            edges[i, 1] = c2
            edges[j, 1] = c1
    return M

def generate_random_matrix(rows=716, cols=179, ones_per_row=8, ones_per_col=32,
                           swaps=100000, rng=None):
    """
    Full pipeline: construct a valid matrix then randomize it with swaps.
    Returns matrix M (dtype uint8).
    """
    M = construct_greedy(rows, cols, ones_per_row, ones_per_col)
    M = randomize_swaps_edge(M, swaps=swaps, rng=rng)
    # sanity checks
    assert np.all(M.sum(axis=1) == ones_per_row), "Row sums incorrect"
    assert np.all(M.sum(axis=0) == ones_per_col), "Column sums incorrect"
    return M


class Model_StaticMapping(nn.Module):
    def __init__(self, seq_num, in_dim, out_dim, hid_dim, num_nodes, tau, random_feature_dim, node_emb_dim, time_emb_dim, \
                 use_residual, use_bn, use_spatial, use_long, dropout, time_of_day_size, day_of_week_size, supports=None, edge_indices=None):
        super(Model_StaticMapping, self).__init__()

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
        self.num_nodes = num_nodes
        
        """
                Mapping in a block ratio of 1:4
        """
        # self.num_dsn_nodes = 179

        # self.dsns_per_node = 1
        # self.nodes_per_dsn = 4
        # self.mapping_from_sensors_to_deep_state_nodes = np.zeros((num_nodes, self.dsns_per_node), dtype=np.int64)
        # self.mapping_from_to_deep_state_nodes_to_sensors = np.zeros((self.num_dsn_nodes, self.nodes_per_dsn), dtype=np.int64)

        # for i in range(num_nodes):
        #     self.mapping_from_sensors_to_deep_state_nodes[i] = [i // 4]

        # for i in range(self.num_dsn_nodes):
        #     self.mapping_from_to_deep_state_nodes_to_sensors[i] = np.arange(4*i, (4*i)+4, dtype = int)

        """
                Mapping identity from sensors to deep state nodes
        """
        # self.num_dsn_nodes = num_nodes

        # self.dsns_per_node = 1
        # self.nodes_per_dsn = 1
        # self.mapping_from_sensors_to_deep_state_nodes = np.zeros((num_nodes, self.dsns_per_node), dtype=np.int64)
        # self.mapping_from_to_deep_state_nodes_to_sensors = np.zeros((self.num_dsn_nodes, self.nodes_per_dsn), dtype=np.int64)

        # for i in range(num_nodes):
        #     self.mapping_from_sensors_to_deep_state_nodes[i] = [i]

        # for i in range(self.num_dsn_nodes):
        #     self.mapping_from_to_deep_state_nodes_to_sensors[i] = [i]
        
        """
                Randomized process that gives for 716 num_nodes the ratio of 32 nodes per dsn and each node has 5 dsns
        """

        self.num_dsn_nodes = 179

        self.dsns_per_node = 2
        self.nodes_per_dsn = 8

        M = generate_random_matrix(self.num_nodes, self.num_dsn_nodes, self.dsns_per_node, self.nodes_per_dsn, swaps=100000)

        self.mapping_from_sensors_to_deep_state_nodes = [list([]) for _ in range(self.num_nodes)]
        self.mapping_from_to_deep_state_nodes_to_sensors = [list([]) for _ in range(self.num_dsn_nodes)]



        for pair in np.argwhere(M == 1):

            nodei, dsnj = pair

            # print(nodei, dsnj)

            self.mapping_from_sensors_to_deep_state_nodes[nodei].append(dsnj)
            self.mapping_from_to_deep_state_nodes_to_sensors[dsnj].append(nodei)


        self.mapping_from_sensors_to_deep_state_nodes = torch.tensor(self.mapping_from_sensors_to_deep_state_nodes, dtype=torch.int64)
        self.mapping_from_to_deep_state_nodes_to_sensors = torch.tensor(self.mapping_from_to_deep_state_nodes_to_sensors, dtype=torch.int64)
        

        self.mapping_from_sensors_to_deep_state_nodes = torch.nn.Parameter(
            torch.tensor(self.mapping_from_sensors_to_deep_state_nodes), requires_grad=False)
        self.mapping_from_to_deep_state_nodes_to_sensors = torch.nn.Parameter(
            torch.tensor(self.mapping_from_to_deep_state_nodes_to_sensors), requires_grad=False)

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

        x = x[:, :, self.mapping_from_to_deep_state_nodes_to_sensors, :] # (B, dim, DSN, pn, 1)
        deep_node_query_vec_tilde = deep_node_query_vec.unsqueeze(2) # (B, DSN, 1 (N), 1, dim)
        obs_node_key_vec = obs_node_key_vec[:, self.mapping_from_to_deep_state_nodes_to_sensors, :] # (B, DSN, pn, 1, dim)


        x = x.transpose(1,2) # (B, DSN, dim, pn, 1)
        x = x.reshape(B * self.num_dsn_nodes, x.shape[2], self.nodes_per_dsn, 1) 

        deep_node_query_vec_tilde = deep_node_query_vec_tilde.reshape(B * self.num_dsn_nodes, 1, 1, deep_node_query_vec_tilde.shape[-1])
        obs_node_key_vec = obs_node_key_vec.reshape(B * self.num_dsn_nodes, self.nodes_per_dsn, 1, obs_node_key_vec.shape[-1])



        x, deep_node_query_vec_tilde_prime, obs_node_key_vec_prime = self.linear_conv[0](x, deep_node_query_vec_tilde, obs_node_key_vec) # (BP, dim * 4, 1, 1)

        x = x.squeeze(2) # removing DSN axis - (BP, dim * 4, 1)
        x = x.reshape(B, self.num_dsn_nodes, x.shape[1], 1) # (B, DSN, dim * 4, pn, 1)
        x = x.transpose(1,2)


        deep_node_key_vec = self.W_3(deep_x_g) # (B, dim, 42, 1)
        deep_node_key_vec = deep_node_key_vec.permute(0, 2, 3, 1) # (B, DSN, 1, dim)


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

        x = self.activation(x) # (B, dim * 4 * 2, DSN, 1)

        obs_node_query_vec = self.W_4(x_g) # (B, dim, N, 1)
        obs_node_query_vec = obs_node_query_vec.permute(0, 2, 3, 1) # (B, N, 1, dim)


        x = x[:, :, self.mapping_from_sensors_to_deep_state_nodes, :] # (B, dim * 4 * 2, N, pdsn, 1)
        obs_node_query_vec = obs_node_query_vec.unsqueeze(2) # (B, N, 1 (DSN), 1, dim)
        deep_node_key_vec = deep_node_key_vec[:, self.mapping_from_sensors_to_deep_state_nodes, :] # (B, N, pdsn, 1, dim)

        x = x.transpose(1,2) # (B, P, dim * 4 * 2, pn, 1)
        x = x.reshape(B * self.num_nodes, x.shape[2], self.dsns_per_node, 1)

        obs_node_query_vec = obs_node_query_vec.reshape(B * self.num_nodes, 1, 1, obs_node_query_vec.shape[-1]) # (B * N, 1 (N), 1, dim)
        deep_node_key_vec = deep_node_key_vec.reshape(B * self.num_nodes, self.dsns_per_node, 1, deep_node_key_vec.shape[-1]) # (B * N, pn, 1, dim)

        x, obs_node_query_vec_prime, deep_node_key_vec_prime = self.linear_conv[-1](x, obs_node_query_vec, deep_node_key_vec) # (BN, dim * 4 * 2, 1, 1)

        x = x.squeeze(2) # removing single nodes axis axis - (BN, dim * 4 * 2, 1)
        x = x.reshape(B, self.num_nodes, x.shape[1], 1)
        x = x.transpose(1,2)

            

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
        
        self.dropout = .3
        self.activation = nn.ReLU()
        self.supports = supports
        
        self.time_num = time_of_day_size
        self.week_num = day_of_week_size

        self.num_contexts = 32
        node_emb_dim = 32
        hid_dim = 128
        
        # node embedding layer
        # self.node_emb_layer = nn.Parameter(torch.empty(num_nodes, node_emb_dim))
        # nn.init.xavier_uniform_(self.node_emb_layer)
        
        self.context_emb_layer = nn.Parameter(torch.empty(self.num_contexts, node_emb_dim))
        nn.init.xavier_uniform_(self.context_emb_layer)
        
        # time embedding layer
        self.time_emb_layer = nn.Parameter(torch.empty(self.time_num, time_emb_dim))
        nn.init.xavier_uniform_(self.time_emb_layer)
        self.week_emb_layer = nn.Parameter(torch.empty(self.week_num, time_emb_dim))
        nn.init.xavier_uniform_(self.week_emb_layer)

        num_values = 3 # number of values in the input sequence (e.g., temperature, humidity, etc.)
        num_context = in_dim - num_values # number of context features (e.g., time, day of week, etc.)

        # embedding layer
        self.input_emb_layer = nn.Conv2d(seq_num * num_values, hid_dim, kernel_size=(1, 1), bias=True)
        self.contextual_emb_layer = nn.Conv2d(num_context, hid_dim, kernel_size=(1, 1), bias=True)

        
        
        self.W_obs_context_key = nn.Conv2d(num_context+time_emb_dim*2, node_emb_dim, kernel_size=(1, 1), bias=True)
        self.W_obs_context_query = nn.Conv2d(num_context+time_emb_dim*2, node_emb_dim, kernel_size=(1, 1), bias=True)
        self.W_1 = nn.Conv2d(node_emb_dim, node_emb_dim, kernel_size=(1, 1), bias=True)
        self.W_2 = nn.Conv2d(node_emb_dim, node_emb_dim, kernel_size=(1, 1), bias=True)
        
        self.linear_conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)
        
        for i in range(self.layer_num):
            self.linear_conv.append(linearized_conv(hid_dim + node_emb_dim, hid_dim + node_emb_dim, self.dropout, self.tau, self.random_feature_dim))
            self.bn.append(nn.LayerNorm(hid_dim + node_emb_dim))


        self.linear_obs_2_dsn_conv = linearized_conv(num_context +  hid_dim + 2 * time_emb_dim, hid_dim, self.dropout, self.tau, self.random_feature_dim)

        self.hid_dim_times_after_conv = 3
        self.linear_dsn_2_obs_conv = linearized_conv(2 * (node_emb_dim + hid_dim), hid_dim * self.hid_dim_times_after_conv, self.dropout, self.tau, self.random_feature_dim)
        
        self.bn_obs_to_context = nn.LayerNorm(hid_dim)
        self.bn_context_to_obs = nn.LayerNorm(hid_dim * self.hid_dim_times_after_conv)
        
        if self.use_long:
            self.regression_layer = nn.Conv2d(hid_dim*4*2+hid_dim+seq_num, out_dim, kernel_size=(1, 1), bias=True)
        else:
            self.regression_layer = nn.Conv2d(hid_dim* (self.hid_dim_times_after_conv + 1) + 2 * time_emb_dim + num_context, out_dim, kernel_size=(1, 1), bias=True)


    def forward(self, x, feat=None):
        
        # x: (B, N, T, D)
        B, N, T, D = x.size()
        
        time_emb = self.time_emb_layer[(x[:, :, -1, 1]*self.time_num).type(torch.LongTensor)]
        week_emb = self.week_emb_layer[(x[:, :, -1, 2]).type(torch.LongTensor)]


        x_context = x[..., -1 , 3:] # shape (B, N, D-3)
        x_value = x[..., :3] # shape (B, N, T, 3)

        # input embedding
        x = x_value.contiguous().view(B, N, -1).transpose(1, 2).unsqueeze(-1) # (B, D*T, N, 1)
        input_emb = self.input_emb_layer(x)

        # context embedding
        x_context = x_context.contiguous().view(B, N, -1).transpose(1, 2).unsqueeze(-1) # (B, D-3, N, 1)
        # input_emb_context = self.contextual_emb_layer(x_context)
        

        # node embeddings
        # node_emb = self.node_emb_layer.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1) # (B, dim, N, 1)

        # time embeddings
        time_emb = time_emb.transpose(1, 2).unsqueeze(-1) # (B, dim, N, 1)
        week_emb = week_emb.transpose(1, 2).unsqueeze(-1) # (B, dim, N, 1)


        
        x_g = torch.cat([x_context, time_emb, week_emb], dim=1) # (B, D-3 +  dim*2, N, 1)
        x = torch.cat([input_emb, x_context, time_emb, week_emb], dim=1) # (B, D-3 + dim*3, N, 1)

        # linearized spatial convolution
        x_pool = [x] # (B,  D-3 + dim*3, N, 1)

        # mapping the node embeddings to the deep state nodes


        # perform normal attention 

        queries = self.context_emb_layer.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1) # (B, dim, C, 1)
        queries_obs = queries.permute(0, 2, 3, 1) # (B, C, 1, dim)
        
        keys = self.W_obs_context_key(x_g) 
        keys = keys.permute(0, 2, 3, 1)# (B, N, 1, dim)



        deepstate, _, _, assignment_scores_source= self.linear_obs_2_dsn_conv(x, queries_obs, keys)

        deepstate = deepstate.permute(0, 2, 3, 1) # (B, C, 1, dim*4)
        deepstate = self.bn_obs_to_context(deepstate)
        deepstate = deepstate.permute(0, 3, 1, 2)

        # merge with the original vector 
        deepstate = torch.concat([deepstate, queries], dim=1) # (B, dim*2, C, 1)

        node_vec1 = self.W_1(queries) # (B, dim, N, 1)
        node_vec2 = self.W_2(queries) # (B, dim, N, 1)
        node_vec1 = node_vec1.permute(0, 2, 3, 1) # (B, N, 1, dim)
        node_vec2 = node_vec2.permute(0, 2, 3, 1) # (B, N, 1, dim)


        deepstate_pool = [deepstate] # (B, C, 1, 2dim)
        for i in range(self.layer_num):
            if self.use_residual:
                residual = deepstate
            deepstate, node_vec1_prime, node_vec2_prime, _ = self.linear_conv[i](deepstate, node_vec1, node_vec2)
            
            if self.use_residual:
                deepstate = deepstate+residual 
                
            if self.use_bn:
                deepstate = deepstate.permute(0, 2, 3, 1) # (B, C, 1, dim*4)
                deepstate = self.bn[i](deepstate)
                deepstate = deepstate.permute(0, 3, 1, 2)

        deepstate_pool.append(deepstate)
        deepstate = torch.cat(deepstate_pool, dim=1) # (B, dim*4, C, 1)
        deepstate = self.activation(deepstate) # (B, dim*4, C, 1)
        
        # mapping the deepstate onto the original nodes


        queries = self.W_obs_context_query(x_g) 
        queries = queries.permute(0, 2, 3, 1)# (B, N, 1, dim)

        keys = self.context_emb_layer.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1) # (B, dim, C, 1)
        keys = keys.permute(0, 2, 3, 1) # (B, C, 1, dim)
       

        x, _, _, assignment_scores_target= self.linear_dsn_2_obs_conv(deepstate, queries, keys)
        
        x = x.permute(0, 2, 3, 1) # (B, C, 1, dim*4)
        x = self.bn_context_to_obs(x)
        x = x.permute(0, 3, 1, 2)
        
        
        
        x_pool.append(x)

        x = torch.cat(x_pool, dim=1) # (B, dim*7 + D - 3, N, 1)
        x = self.activation(x)


        
        if self.use_long:
            feat = feat.permute(0, 2, 1).unsqueeze(-1) # (B, F, N, 1)
            x = torch.cat([x, feat], dim=1)
            x = self.regression_layer(x) # (B, N, T)
            x = x.squeeze(-1).permute(0, 2, 1)
        else:
            x = self.regression_layer(x) # (B, N, T)
            x = x.squeeze(-1).permute(0, 2, 1)

        return {"prediction": x.transpose(1,2).unsqueeze(-1)
              , "node_vec1": node_vec1_prime
              , "node_vec2": node_vec2_prime
              , "supports": self.supports
              , 'use_spatial': self.use_spatial 
              , 'deep_node_vec': self.context_emb_layer
              , 'assignment_scores_source': assignment_scores_source
              , 'assignment_scores_target': assignment_scores_target}
