import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
"""
Start of utilitiy methods that were copied from BigST
"""

from src.base.model import BaseModel

def create_products_of_givens_rotations(dim, seed):
    nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
    q = np.eye(dim, dim)
    np.random.seed(seed)
    for _ in range(nb_givens_rotations):
        random_angle = math.pi * np.random.uniform()
        random_indices = np.random.choice(dim, 2)
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])
        slice_i = q[index_i]
        slice_j = q[index_j]
        new_slice_i = math.cos(random_angle) * slice_i + math.cos(random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
        q[index_i] = new_slice_i
        q[index_j] = new_slice_j
    return torch.tensor(q, dtype=torch.float32)

def create_random_matrix(m, d, seed:int|torch.Tensor=0, scaling=0, struct_mode=False):
    nb_full_blocks = int(m/d)
    block_list = []
    current_seed = seed
    for _ in range(nb_full_blocks):
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.linalg.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q)
        current_seed += 1
    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.linalg.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q[0:remaining_rows])
    final_matrix = torch.vstack(block_list)

    current_seed += 1
    torch.manual_seed(current_seed)
    if scaling == 0:
        multiplier = torch.norm(torch.randn((m, d)), dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
    else:
        raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

    return torch.matmul(torch.diag(multiplier), final_matrix)

def random_feature_map(data, is_query, projection_matrix, numerical_stabilizer=0.000001):
    data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32)))
    data = data_normalizer * data
    ratio = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32))
    data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)
    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=len(data.shape)-1)
    diag_data = diag_data / 2.0
    diag_data = torch.unsqueeze(diag_data, dim=len(data.shape)-1)
    last_dims_t = len(data_dash.shape) - 1
    attention_dims_t = len(data_dash.shape) - 3
    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash, dim=last_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
                    dim=attention_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )
    return data_dash

def linear_kernel(x, node_vec1, node_vec2, filter_mat):
    # x: [B, N, 1, nhid] node_vec1: [B, N, 1, r], node_vec2: [B, N, 1, r]
    # sum of k_m * v_m
    if filter_mat is not None:
        v2x = torch.einsum("bnhm,bnhd->nbhmd", node_vec2, x)
        v2x = F.softmax(v2x, dim = -2)
        # sum over all N the keys * values = sum of k_m * v_m
        v2x = torch.einsum("nbhmd,nc->cbhmd", v2x, filter_mat) 
        # q_t * sum^T_m k_m * v_m
        out1 = torch.einsum("bchm,cbhmd->bchd", node_vec1, v2x) # [N, B, 1, nhid]
    else: 
        v2x = torch.einsum("bnhm,bnhd->bhmd", node_vec2, x)
        v2x = F.softmax(v2x, dim = -2)
        out1 = torch.einsum("bnhm,bhmd->bnhd", node_vec1, v2x) # [N, B, 1, nhid]

    # one_matrix = torch.ones([node_vec2.shape[0]]).to(node_vec1.device)
    # sum over all N the keys = sum of k_m
    # node_vec2_sum = torch.einsum("nbhm,n->bhm", node_vec2, one_matrix)
    node_vec2_sum = torch.sum(node_vec2, dim = 1)
    # q_t * sum^T_m k_m
    out2 = torch.einsum("bnhm,bhm->bnh", node_vec1, node_vec2_sum) # [N, 1]

    out2 = torch.unsqueeze(out2, len(out2.shape))
    out = out1 #/ out2 # [B, N, 1, nhid]

    return out, out2


class conv_approximation(nn.Module):
    def __init__(self, dropout, tau, random_feature_dim):
        super().__init__()
        self.tau = tau
        self.random_feature_dim = random_feature_dim
        self.register_buffer("projection_matrix", None, persistent=False)
        self.register_buffer("inv_sqrt_tau", torch.tensor(1.0 / math.sqrt(tau)), persistent=False)

    def _maybe_init_projection(self, dim, device, dtype):
        need_new = (
            self.projection_matrix is None
            or self.projection_matrix.shape != (self.random_feature_dim, dim)
            or self.projection_matrix.device != device
            or self.projection_matrix.dtype != dtype
        )
        if need_new:
            with torch.no_grad():
                q = torch.randn((dim, dim), device=device, dtype=dtype)
                q = torch.linalg.qr(q, mode="reduced").Q.T
                if self.random_feature_dim <= dim:
                    proj = q[:self.random_feature_dim]
                else:
                    repeat = (self.random_feature_dim + dim - 1) // dim
                    proj = q.repeat(repeat, 1)[:self.random_feature_dim]
            self.projection_matrix = proj

    def forward(self, x, node_vec1, node_vec2, filter_mat):
        # dim = node_vec1.shape[-1]
        # self._maybe_init_projection(dim, node_vec1.device, node_vec1.dtype)

        # node_vec1 = node_vec1 * self.inv_sqrt_tau
        # node_vec2 = node_vec2 * self.inv_sqrt_tau
        node_vec1_prime = None #random_feature_map(node_vec1, True, self.projection_matrix)
        node_vec2_prime = None #random_feature_map(node_vec2, False, self.projection_matrix)
        x, D = linear_kernel(x, node_vec1, node_vec2, filter_mat)
        return x, node_vec1_prime, node_vec2_prime, D

class linearized_conv(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout, tau=1.0, random_feature_dim=64, non_linearity = True, key_dim = None):
        super(linearized_conv, self).__init__()
        
        self.dropout = dropout
        self.tau = tau
        self.random_feature_dim = random_feature_dim
        self.non_linearity = non_linearity  
        
        self.input_fc = nn.Conv2d(in_channels=in_dim, out_channels=hid_dim, kernel_size=(1, 1), bias=True)
        self.output_fc = nn.Conv2d(in_channels=key_dim, out_channels=hid_dim, kernel_size=(1, 1), bias=True)
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout)
        
        self.conv_app_layer = conv_approximation(self.dropout, self.tau, self.random_feature_dim)
        
        self.attention_layer = nn.MultiheadAttention(embed_dim=key_dim, num_heads=1, dropout=dropout, batch_first=True, vdim=hid_dim, kdim=key_dim)
        
    def forward(self, input_data, node_vec1, node_vec2, filter_mat):
        x = self.input_fc(input_data)
        
        if self.non_linearity:
            x = self.activation(x)
            x = self.dropout_layer(x)
        
        x = x.permute(0, 2, 3, 1) # (B, N, 1, dim*4)
        x, node_vec1_prime, node_vec2_prime, D = self.conv_app_layer(x, node_vec1, node_vec2, filter_mat)
        
        # x = x.squeeze(2) # (B, N, dim*4)
        # node_vec1 = node_vec1.squeeze(2) # (B, N, dim)
        # node_vec2 = node_vec2.squeeze(2) # (B, N, dim)
        
        # if filter_mat is None:
        #     x, attn_weights = self.attention_layer(node_vec1, node_vec2, x, need_weights=True) # (B, N, dim)
        # else:
            
        #     filter_mat_attention  = torch.zeros_like(filter_mat, dtype = torch.bool)
        #     filter_mat_attention[filter_mat < 1e-8] = True
        #     x, attn_weights =  self.attention_layer(node_vec1, node_vec2, x, need_weights=True, attn_mask=filter_mat_attention.T) # (B, N, dim)
                
        # x = x.unsqueeze(2) # (B, N, 1, dim)
        
        x = x.permute(0, 3, 1, 2) # (B, dim*4, N, 1)
        
        # x = self.output_fc(x) # (B, dim, N, 1)
        
        return x, node_vec1_prime, node_vec2_prime, D

"""
End of utilitiy methods that were copied from BigST
"""

class DeepStateGNN(BaseModel):

    def __init__(self, num_nodes, in_dim, out_dim, random_feature_dim,
                 time_emb_dim, seq_num, node_emb_dim, use_spatial, dropout,
                 n_contexts,hid_dim,
                 time_of_day_size=288, day_of_week_size=7,
                 use_residual=True, use_bn=True, layer_num=3, adding_query_to_dsn = True):
        super(DeepStateGNN, self).__init__(num_nodes, in_dim, out_dim)

        self.tau = .25
        self.layer_num = layer_num
        self.in_dim = in_dim
        self.random_feature_dim = random_feature_dim
        
        self.use_residual = use_residual
        self.use_bn = use_bn
        
        self.dropout = dropout
        self.activation = nn.ReLU()
        
        self.time_num = time_of_day_size
        self.week_num = day_of_week_size

        self.num_contexts = n_contexts
        self.node_emb_dim = node_emb_dim

        self.use_spatial = use_spatial
        self.adding_query_to_dsn = adding_query_to_dsn

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
        self.input_emb_layer = nn.Conv2d(seq_num * num_values, hid_dim, kernel_size=(1, 1), bias=False)


        # Use a gating mechnism to map the observations to deep state nodes
        self.context_processing = nn.Sequential(
            nn.Conv2d(num_context , hid_dim, kernel_size=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, 1), bias=False),
            # nn.ReLU(),
            # nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, 1), bias=False),
            # nn.ReLU(),
            # nn.Tanh(),
        )
        
        self.W_obs_context_key = nn.Conv2d(hid_dim+time_emb_dim*2, node_emb_dim, kernel_size=(1, 1), bias=False)
        self.W_obs_context_query = nn.Conv2d(hid_dim+time_emb_dim*2, node_emb_dim, kernel_size=(1, 1), bias=False)
        self.W_1 = nn.Conv2d(node_emb_dim, node_emb_dim, kernel_size=(1, 1), bias=True)
        self.W_2 = nn.Conv2d(node_emb_dim, node_emb_dim, kernel_size=(1, 1), bias=True)
        
        self.linear_conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        
        for _ in range(self.layer_num):
            self.linear_conv.append(linearized_conv(hid_dim + node_emb_dim, hid_dim + node_emb_dim, self.dropout, self.tau, self.random_feature_dim, non_linearity=False, key_dim = node_emb_dim))
            self.bn.append(nn.LayerNorm(hid_dim + node_emb_dim))
            
        


        self.linear_obs_2_dsn_conv = linearized_conv(hid_dim  + 2 * time_emb_dim, hid_dim, self.dropout, self.tau, self.random_feature_dim, non_linearity=True, key_dim = node_emb_dim)

        self.hid_dim_times_after_conv = 1

        # self.W_in = nn.Conv2d(num_context +  hid_dim + 2 * time_emb_dim, hid_dim, kernel_size=(1, 1), bias=True)
        # self.W_out = nn.Conv2d(2 * (node_emb_dim + hid_dim), hid_dim * self.hid_dim_times_after_conv, kernel_size=(1, 1), bias=True)

        self.linear_dsn_2_obs_conv = linearized_conv( (node_emb_dim + hid_dim) * 2, hid_dim * self.hid_dim_times_after_conv, self.dropout, self.tau, self.random_feature_dim, key_dim = node_emb_dim)
        
        self.bn_obs_to_context = nn.LayerNorm(hid_dim)
        self.bn_context_to_obs = nn.LayerNorm(hid_dim * self.hid_dim_times_after_conv)
        
        self.regression_layer = nn.Conv2d(hid_dim* (self.hid_dim_times_after_conv + 1) + 2 * time_emb_dim , out_dim, kernel_size=(1, 1), bias=True)

    def forward(self, x, feat=None, static_prefilter = None, valid_observations = None, query_index = None):       
        # x: (B, N, T, D)
        B, N, T, D = x.size()
        
        time_emb = self.time_emb_layer[(x[:, :, -1, 1]*self.time_num).int()]
        # TODO: why isn't week values multiplied by week_num first?
        week_emb = self.week_emb_layer[x[:, :, -1, 2].int()]


        x_context = x[..., -1 , 3:] # shape (B, N, D-3)
        x_value = x[..., :3] # shape (B, N, T, 3)
        
        assert torch.any(torch.isnan(x_value)) == False, "Input contains NaN values"

        # input embedding
        x = x_value.contiguous().view(B, N, -1).transpose(1, 2).unsqueeze(-1) # (B, D*T, N, 1)
        input_emb = self.input_emb_layer(x)
        # context embedding
        x_context = x_context.contiguous().view(B, N, -1).transpose(1, 2).unsqueeze(-1) # (B, D-3, N, 1)
        x_context = self.context_processing(x_context) # (B, D-3, N, 1)
        # time embeddings
        time_emb = time_emb.transpose(1, 2).unsqueeze(-1) # (B, dim, N, 1)
        week_emb = week_emb.transpose(1, 2).unsqueeze(-1) # (B, dim, N, 1)

        x_g = torch.cat([x_context, time_emb, week_emb], dim=1) # (B, D-3 +  dim*2, N, 1)
        x = torch.cat([input_emb + x_context, time_emb, week_emb], dim=1) # (B, D-3 + dim*3, N, 1)

        # linearized spatial convolution
        if query_index is not None:
            x_pool = [x[:,:, query_index:query_index+1, :]] # (B,  D-3 + dim*3, 1, 1)
        else:
            x_pool = [x] # (B,  D-3 + dim*3, N, 1)

        # mapping the node embeddings to the deep state nodes
        # q: dsn states | keys: observation embeddings
        queries = self.context_emb_layer.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1) # (B, dim, C, 1)
        
        raw_dsn = queries.permute(0, 2, 3, 1).squeeze(2) # (B, C, dim)
        queries_obs = queries.permute(0, 2, 3, 1) # (B, C, 1, dim)
        
        keys = self.W_obs_context_key(x_g) 
        keys = keys.permute(0, 2, 3, 1)# (B, N, 1, dim)

        if static_prefilter is not None:

            # deepstate = self.W_in(x)
            # assignment_scores_source = None
            deepstate, _, _, assignment_scores_source= self.linear_obs_2_dsn_conv(x, queries_obs, keys, static_prefilter)
        else:
            deepstate, _, _, assignment_scores_source= self.linear_obs_2_dsn_conv(x, queries_obs, keys, None)

        # deepstate: weighted observations combined
        deepstate = deepstate.permute(0, 2, 3, 1) # (B, C, 1, dim*4)
        deepstate = self.bn_obs_to_context(deepstate)
        deepstate = deepstate.permute(0, 3, 1, 2)

        obs_augmented_dsn = deepstate.permute(0, 2, 1, 3).squeeze(-1)  # (B, C, dim*4)


        # merge with the original vector 
        # deepstate: concatenated combined observations with the original deepstate embeddings
        node_vec1 = self.W_1(queries) # (B, dim, N, 1)
        node_vec2 = self.W_2(queries) # (B, dim, N, 1)
        node_vec1 = node_vec1.permute(0, 2, 3, 1) # (B, N, 1, dim)
        node_vec2 = node_vec2.permute(0, 2, 3, 1) # (B, N, 1, dim)

        if self.adding_query_to_dsn:
            deepstate = torch.concat([deepstate, queries], dim=1) # (B, dim*2, C, 1)
        else:
            deepstate = torch.concat([deepstate, torch.zeros_like(queries)], dim=1) # (B, dim*2, C, 1)

        # perform several layers of graph convolution on the deep state nodes
        #### Self attentiopn between DSN states begin
        


        deepstate_pool = [deepstate] # (B, C, 1, 2dim)
        for i in range(self.layer_num):
            if self.use_residual:
                residual = deepstate
            deepstate, node_vec1_prime, node_vec2_prime, _ = self.linear_conv[i](deepstate, node_vec1, node_vec2, None)
            
            if self.use_residual:
                deepstate = deepstate+residual 
                
            if self.use_bn:
                deepstate = deepstate.permute(0, 2, 3, 1) # (B, C, 1, dim*4)
                deepstate = self.bn[i](deepstate)
                deepstate = deepstate.permute(0, 3, 1, 2)

        deepstate_pool.append(deepstate)
        deepstate = torch.cat(deepstate_pool, dim=1) # (B, dim*4, C, 1)
        # deepstate = self.activation(deepstate) # (B, dim*4, C, 1)
        #### Self attentiopn between DSN states end

        gnn_convolved_dsn = deepstate.permute(0, 2, 1, 3).squeeze(-1)  # (B, C, dim*4)
        


        # mapping the deepstate onto the original nodes
        # from here now on it's the inverse. Basically from the DSN states and the context of the observations, reconstruct the original traffic features
        # queries: embedding of the observation context
        if query_index is not None:
            x_g = x_g[:,:, query_index:query_index+1, :] # (B, 1, N, 1)
            static_prefilter = static_prefilter[ query_index:query_index+1, :] # (N, 1)
        queries = self.W_obs_context_query(x_g) 
        queries = queries.permute(0, 2, 3, 1)# (B, N, 1, dim)

        # keys are the DSN states after the self-attention
        keys = self.context_emb_layer.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1) # (B, dim, C, 1)
        keys = keys.permute(0, 2, 3, 1) # (B, C, 1, dim)

        if static_prefilter is not None:
            # x_org, _, _, assignment_scores_target= self.linear_dsn_2_obs_conv(deepstate_pool[0], queries, keys, static_prefilter.T)  
            # x_conv, _, _, assignment_scores_target= self.linear_dsn_2_obs_conv(deepstate_pool[1], queries, keys, static_prefilter.T)  
            x, _, _, assignment_scores_target= self.linear_dsn_2_obs_conv(deepstate, queries, keys, static_prefilter.T)
            # x = self.W_out(deepstate)
            # assignment_scores_target = None
            
            x = x.permute(0, 2, 3, 1) # (B, C, 1, dim*4)
            # split the x into half in the last dimesion and add both halfs up
            
            # x = torch.split(x, x.shape[-1] // 2, dim=3)            
            # x = x[0] + x[1] # (B, C, 1, dim*4)
            
            x = self.bn_context_to_obs(x)
            x = x.permute(0, 3, 1, 2)
        else:
            
            x, _, _, assignment_scores_target= self.linear_dsn_2_obs_conv(deepstate, queries, keys, None)
            
            # x_org, _, _, assignment_scores_target= self.linear_dsn_2_obs_conv(deepstate_pool[0], queries, keys, None)
            # x_conv, _, _, assignment_scores_target= self.linear_dsn_2_obs_conv(deepstate_pool[1], queries, keys, None)

        
            # x_org = x_org.permute(0, 2, 3, 1) # (B, C, 1, dim*4)
            # x_org = self.bn_context_to_obs(x_org)
            # x_org = x_org.permute(0, 3, 1, 2)
            
            # x_conv = x_conv.permute(0, 2, 3, 1) # (B, C, 1, dim*4)
            # x_conv = self.bn_context_to_obs(x_conv)
            # x_conv = x_conv.permute(0, 3, 1, 2)
        
            x = x.permute(0, 2, 3, 1) # (B, C, 1, dim*4)
            x = self.bn_context_to_obs(x)
            x = x.permute(0, 3, 1, 2)


    #### Inverse ends
        
        
        #### Here is from BigST for a 1-1 mapping from nodes to the traffic features
        x_pool.append(x) # + x_org) # (B, dim*4 + dim*4, N, 1)
        x = torch.cat(x_pool, dim=1) # (B, dim*7 + D - 3, N, 1)
        x = self.activation(x)

        x = self.regression_layer(x) # (B, N, T)
        x = x.squeeze(-1).permute(0, 2, 1)

        # x = self.restore_input_shape(x, (B, N_org, T), label_mask)

        return {"prediction": x.transpose(1,2).unsqueeze(-1)
              , 'assignment_scores_source': assignment_scores_source
              , 'assignment_scores_target': assignment_scores_target
              , 'dsn_states': {
                    'raw': raw_dsn,
                    'obs_augmented': obs_augmented_dsn,
                    'gnn_convolved': gnn_convolved_dsn
                    }
}

