import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import numpy as np
from tqdm import tqdm

from src.base.model import BaseModel
from torch.profiler import record_function


class ObservationEmbeddingModule(nn.Module):

    def __init__(self, num_traffic_features, dynamic_feature_indices, static_feature_indices,
                 nodes_hidden_dim, num_gru_layers, num_hidden_mlp_layers, observation_hidden_dim,
                 context_in_embedding) -> None:
        super().__init__()
        self.num_traffic_features = num_traffic_features
        self.dynamic_feature_indices = dynamic_feature_indices
        self.static_feature_indices = static_feature_indices
        self.context_in_embedding = context_in_embedding
        self.gru_hidden_dim = observation_hidden_dim

        self.traffic_values_gru = nn.GRU(input_size = self.num_traffic_features, hidden_size = self.gru_hidden_dim, num_layers = num_gru_layers,  batch_first= False)
        self.dynamic_features_gru = nn.GRU(input_size = len(self.dynamic_feature_indices), hidden_size = self.gru_hidden_dim, num_layers = num_gru_layers,  batch_first= False)
        
        obs_mlp_input_size = self.gru_hidden_dim
        if self.context_in_embedding:
            obs_mlp_input_size += observation_hidden_dim 

        self.obervation_mlp = nn.Sequential(
            nn.Linear(obs_mlp_input_size , observation_hidden_dim),
            nn.ReLU(),
            nn.Linear(observation_hidden_dim, observation_hidden_dim),
            nn.ReLU(),
            nn.Linear(observation_hidden_dim, observation_hidden_dim),
            nn.Tanh()
        )


        self.intermediate_embedding = nn.Sequential(
            nn.Linear(self.gru_hidden_dim + len(static_feature_indices), observation_hidden_dim),
            *[nn.ReLU(), nn.Linear(observation_hidden_dim, observation_hidden_dim)] * num_hidden_mlp_layers,
            nn.ReLU()
            )
        
        self.query_embedding_mlp = nn.Sequential(
            nn.Linear(observation_hidden_dim, observation_hidden_dim),
            nn.ReLU(),
            nn.Linear(observation_hidden_dim, nodes_hidden_dim),
            # Use tanh to map the values between -1 and 1. The sum of multiple embedding can still center arounf zero.
            nn.Tanh()
        )

        self.query_layer_norm = nn.LayerNorm(nodes_hidden_dim)
        self.observation_layer_norm = nn.LayerNorm(observation_hidden_dim)

    def _intermediate_embedding(self, x):
        # Taking the dynamic features and the static features to create an intermediate embedding that will be topped with an extra MLP to create the query embedding or the observation embedding
        dynamic_features = x[..., self.dynamic_feature_indices]
        _, hidden_state_dynamic_features = self.dynamic_features_gru(dynamic_features)

        non_traffic_values_first_row = x[0, :, self.static_feature_indices]

        input_to_mlp = torch.concat([hidden_state_dynamic_features[-1], non_traffic_values_first_row], dim = -1)

        return self.intermediate_embedding(input_to_mlp)

    def query_embedding(self, x):
        # Using the intermeadiate embedding from the dynamic features and the static features to create the query embedding with an extra MLP
        intermediate_embedding = self._intermediate_embedding(x)
        query_embed = self.query_embedding_mlp(intermediate_embedding)
        return self.query_layer_norm(query_embed)
    

    def embed_traffic_values(self, traffic_vals_tensor):
        _, hidden_state_traffic = self.traffic_values_gru(traffic_vals_tensor)
        return hidden_state_traffic[-1]
    

    def forward(self, x):
        # The input shape of x is (num_timestamps, batch_size * num_per_obs, num_features)
        # Output will be (batch_size * num_per_obs, num_features)

        # Take the traffic values to embed them in a GRU
        traffic_values = x[..., :self.num_traffic_features]
        _, hidden_state_traffic = self.traffic_values_gru(traffic_values) 
        
        if self.context_in_embedding:
            intermediate_embedding = self._intermediate_embedding(x)
            input_to_mlp = torch.concat([hidden_state_traffic[-1], intermediate_embedding], dim = -1)
        else:
            input_to_mlp = hidden_state_traffic[-1]

        observation_embed = self.obervation_mlp(input_to_mlp)
        return self.observation_layer_norm(observation_embed), hidden_state_traffic[-1]
        


class ClusterAssignmentModule(nn.Module):
    """This is the class to give soft assignment scores from observations to metanodes.
    During the forward pass, it takes the metanode states as keys and the observations as queries and returns the soft assignment scores. 
    """

    def __init__(self, num_attention_heads, nodes_hidden_dim, considered_features_indices, reconstruction_horizon) -> None:
        super().__init__()
        self.considered_features_indices = considered_features_indices
        self.query_gru = nn.GRU(input_size = len(considered_features_indices), hidden_size = nodes_hidden_dim, num_layers = 1,  batch_first= False)
        self.multi_head_attention = nn.MultiheadAttention(nodes_hidden_dim, num_attention_heads, batch_first= True)
        self.reconstruction_horizon = reconstruction_horizon

        self.embedding_module = nn.Sequential(
            nn.Linear(len(considered_features_indices) * self.reconstruction_horizon, nodes_hidden_dim),
            nn.ReLU(),
            nn.Linear(nodes_hidden_dim, nodes_hidden_dim),
        )

        self.gate = nn.Sequential(
            nn.Linear(nodes_hidden_dim + len(considered_features_indices) * self.reconstruction_horizon, max(nodes_hidden_dim , len(considered_features_indices) * self.reconstruction_horizon)),
            nn.ReLU(),
            nn.Linear(max(nodes_hidden_dim , len(considered_features_indices) * self.reconstruction_horizon), nodes_hidden_dim),
            nn.ReLU(),
            nn.Linear(nodes_hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, metanode_states, observations):
        # metanode_states shape is (num_metanodes, nodes_hidden_dim)
        # observations shape is (num_timesteps, num_observations, len(considered_features_indices))
        target_observations = observations[...,  self.considered_features_indices]
        # get the keys and values from the metanode states
        # the shape of keys and values is (num_metanodes, nodes_hidden_dim)
        # keys = metanode_states.detach().clone()
        # values = metanode_states.detach().clone()


        # pass the queries through a GRU layer
        # the shape of queries_seq_embedding is (1, num_observations, nodes_hidden_dim)
        # _, queries_seq_embedding = self.query_gru(target_observations)
        # Take the last layer hidden state as the query
        # queries_seq_embedding = queries_seq_embedding[-1, :, :]
        # queries shape will be (num_observations, nodes_hidden_dim)
        # queries = queries_seq_embedding
        # pass the queries, keys, and values through a multihead attention layer
        # the shape of assignment_scores is (num_observations, num_metanodes)
        # assignment_scores = torch.softmax(assignment_scores, dim=-1)
        
        
        # queries = self.embedding_module(target_observations.transpose(0,1).reshape(-1, len(self.considered_features_indices) * self.reconstruction_horizon))
        # _, assignment_scores = self.multi_head_attention(queries, keys, values, need_weights=True)

        flat_observations = target_observations.reshape(-1, len(self.considered_features_indices) * self.reconstruction_horizon).unsqueeze(1).expand(-1, metanode_states.shape[0], -1)
        # concatetante the meta node states with the flat observations
        dsn_states = metanode_states.unsqueeze(0).expand(flat_observations.shape[0], -1, -1)


        gate_input = torch.cat((dsn_states, flat_observations), dim = -1)

        gate = self.gate(gate_input)

        return gate.squeeze(-1)

        # return assignment_scores


class BatchAttBiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_gru_layers, attention_hidden_dim):
        super(BatchAttBiGRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers
        self.attention_hidden_dim = attention_hidden_dim
        
        # Bidirectional GRU layer
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, bidirectional=True, batch_first=True, num_layers=self.num_gru_layers, bias=False)
        
        # Attention layer
        self.attn = nn.Linear(self.hidden_dim * 2, self.attention_hidden_dim)
        self.v = nn.Linear(self.attention_hidden_dim, 1, bias=False)

        # define a layer to tranform the hidden states to the same dimension as the attention_hidden_dim

        self.hidden_transform = nn.Linear(self.hidden_dim * 2, self.attention_hidden_dim, bias = False)

        # Layer normalization

        self.layer_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x, assignment_scores):

        # x is of shape (num_metanodes, num_observations, hidden_dim)
        # sort the x based on the assignment scores 
        x = torch.gather(x, 1,torch.argsort(assignment_scores, dim = 0).T[..., None].expand( (-1,-1,self.input_dim)))

        # thresholded_assigment_scores is (#num_observations, #num_metanodes) and we need the number of zeros for each meta node
        zeros_per_metanode = torch.sum(assignment_scores < 1e-5, dim=0)


        # x of shape (num_metas, num_observations, hidden_dim) 
        # zeros_per_metanode is of shape (num_metas)

        out, _ = self.gru(x) # out shape: (num_metas, num_observations, hidden_dim * 2)

        last_hidden_forward = out[:, -1, :self.hidden_dim] # shape: (num_metas, hidden_dim) # this is true because x is sorted with the assignment scores increasing
        # last_hidden_backward = out[:, 0, self.hidden_dim:] # shape: (num_metas, hidden_dim)
        torch.minimum(zeros_per_metanode, torch.full_like(zeros_per_metanode, fill_value = out.shape[1] - 1), out = zeros_per_metanode) # If a row is completely empty, then the last hidden state is the first hidden state whhich is a zero. So add the end the attn is equals the bias of the attn layer.
        last_hidden_backward = torch.gather(out, 1, zeros_per_metanode.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.hidden_dim * 2))[:, 0, self.hidden_dim:]

        # aggregated last hidden states with max pooling
        aggregated_last_hidden = torch.tanh(self.hidden_transform(torch.cat((last_hidden_forward, last_hidden_backward), dim=-1))) # shape: (num_metas, attention_hidden_dim)

        # alignment vectors for each observation is the hidden state of the previous timestamp in the forward direction
        padded_out = torch.cat((torch.zeros(out.shape[0], 1, self.hidden_dim * 2, device=x.device), out, torch.zeros(out.shape[0], 1, self.hidden_dim * 2, device=x.device)), dim=1) # shape: (batch_size, seq_len + 2, hidden_dim * 2)
        alignment_vectors = torch.cat((padded_out[:, :-2, :self.hidden_dim], padded_out[:, 2:, self.hidden_dim:]), dim=-1) # shape: (batch_size, seq_len, hidden_dim * 2)
        alignment_vectors = torch.tanh(self.hidden_transform(alignment_vectors)) # shape: (batch_size, seq_len, attention_hidden_dim)

        # create a mask with the shape batch_size, seq_len to mask out those padded values that are within the zeros_per_metanode
        mask = torch.arange(alignment_vectors.shape[1], device = alignment_vectors.device).unsqueeze(0) < zeros_per_metanode.unsqueeze(-1) # shape: (batch_size, seq_len)
        mask = mask.unsqueeze(-1).expand(-1, -1, alignment_vectors.shape[-1]) # shape: (batch_size, seq_len, attention_hidden_dim)
        alignment_vectors = alignment_vectors * mask

        # align the aggregated last hidden states with each hidden state (output of the GRU) to get the attention scores for each observation
        # shape: (batch_size, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((alignment_vectors, aggregated_last_hidden.unsqueeze(1).expand(-1, alignment_vectors.shape[1], -1)), dim=-1)))
        neg_attention_scores = self.v(energy).squeeze(-1) # shape: (batch_size, seq_len)
        # apply softmax to get the attention scores
        attention_scores = F.softmax(-neg_attention_scores, dim=-1) # shape: (batch_size, seq_len)
        # attention_scores = torch.ones_like(neg_attention_scores) - neg_attention_scores

        # apply the attention scores to the hidden states
        # shape: (batch_size, hidden_dim)
        out = torch.tanh(self.hidden_transform(out)) # shape: (batch_size, seq_len, hidden_dim)
        context_matrix = torch.einsum('bse,bs->be', out, attention_scores)

        return context_matrix
        

class Attentional_BiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_gru_layers, attention_hidden_dim):
        super(Attentional_BiGRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers
        self.attention_hidden_dim = attention_hidden_dim
        
        # Bidirectional GRU layer
        self.gru = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True, num_layers=num_gru_layers)
        
        # Attention layer
        self.attn = nn.Linear(hidden_dim * 2, attention_hidden_dim)
        self.v = nn.Linear(attention_hidden_dim, 1, bias=False)

        # define a layer to tranform the hidden states to the same dimension as the attention_hidden_dim
        # TODO: is using this a good idea? The forward and backward hidden states are so different so this is 
        # why I used this hidden_transform. Max would only pick one of the hidden states.
        self.hidden_transform = nn.Linear(hidden_dim * 2, attention_hidden_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)


    def forward(self, x):
        # x is of shape (num_metanodes, num_observations, hidden_dim)
        # num_metanodes can be treated as the batch size, num_observations can be treated as the sequence length.
        # Filter out zero vector observations from the input x, get a new view of x with the non-zero vectors only.
        non_zero_observations_mask = torch.any(x != 0, dim=-1)
        context_matrix = torch.empty(x.shape[0], self.hidden_dim, device=x.device)
        for i in range(x.shape[0]):
            # filtered_x shape: (non_zero_observations for the medanode, hidden_dim)
            filtered_x = x[i, non_zero_observations_mask[i], :]

            # if filtered_x is empty, then the context vector is a zero vector
            if filtered_x.shape[0] == 0:
                context_matrix[i] = torch.zeros(self.hidden_dim, device=x.device)
                continue

            out, _ = self.gru(filtered_x.unsqueeze(0)) # out shape: (1, seq_len, hidden_dim * 2)
            # if only one observation is available, tr
        
            last_hidden_forward = out[:, -1, :self.hidden_dim] # shape: (batch_size, hidden_dim)
            last_hidden_backward = out[:, 0, self.hidden_dim:] # shape: (batch_size, hidden_dim)

            # aggregated last hidden states with max pooling
            # aggregated_last_hidden = self.hidden_aggregation_func(last_hidden_forward, last_hidden_backward) # shape: (batch_size, hidden_dim)
            aggregated_last_hidden = torch.tanh(self.hidden_transform(torch.cat((last_hidden_forward, last_hidden_backward), dim=-1))) # shape: (batch_size, attention_hidden_dim)
            # alignment vectors for each observation is the hidden state of the previous timestamp in the forward direction 
            # concatenated with the hidden state of the next timestamp in the backward direction
            # For the first timestamp, the forward hidden state of previous timestamp is zero vector.
            # for the last timestamp, the backward hidden state of the next timestamp is zero vector.
            padded_out = torch.cat((torch.zeros(out.shape[0], 1, self.hidden_dim * 2, device=x.device), out, torch.zeros(out.shape[0], 1, self.hidden_dim * 2, device=x.device)), dim=1) # shape: (batch_size, seq_len + 2, hidden_dim * 2)
            alignment_vectors = torch.cat((padded_out[:, :-2, :self.hidden_dim], padded_out[:, 2:, self.hidden_dim:]), dim=-1) # shape: (batch_size, seq_len, hidden_dim * 2)
            alignment_vectors = torch.tanh(self.hidden_transform(alignment_vectors)) # shape: (batch_size, seq_len, attention_hidden_dim)

            # align the aggregated last hidden states with each hidden state (output of the GRU) to get the attention scores for each observation
            # shape: (batch_size, seq_len, 1)
            energy = torch.tanh(self.attn(torch.cat((alignment_vectors, aggregated_last_hidden.unsqueeze(1).expand(-1, alignment_vectors.shape[1], -1)), dim=-1)))
            neg_attention_scores = self.v(energy).squeeze(-1) # shape: (batch_size, seq_len)
            # apply softmax to get the attention scores
            attention_scores = F.softmax(-neg_attention_scores, dim=-1) # shape: (batch_size, seq_len)
            # attention_scores = torch.ones_like(neg_attention_scores) - neg_attention_scores

            # apply the attention scores to the hidden states
            # shape: (batch_size, hidden_dim)
            out = torch.tanh(self.hidden_transform(out)) # shape: (batch_size, seq_len, hidden_dim)
            context_vector = torch.einsum('bse,bs->be', out, attention_scores)

            context_vector = self.layer_norm(context_vector)
            context_matrix[i] = context_vector
        return context_matrix
    

class MetanodeGraphEmbedding(nn.Module):

    def __init__(self, graph_embedding_dim, num_metanodes, nodes_hidden_dim) -> None:
        super().__init__()
        self.graph_embedding_dim = graph_embedding_dim
        self.num_metanodes = num_metanodes
        self.nodes_hidden_dim = nodes_hidden_dim


        self.mlp = nn.Sequential(
            nn.Linear(self.num_metanodes * self.nodes_hidden_dim, (self.num_metanodes * self.nodes_hidden_dim) // 2),
            nn.ReLU(),
            nn.Linear((self.num_metanodes * self.nodes_hidden_dim) // 2, graph_embedding_dim),
            nn.ReLU(),
            nn.Linear(graph_embedding_dim, graph_embedding_dim),
            nn.Tanh(),
        )

    def forward(self, x):

        # Embed the node features from the traffic reconstruction graph into a hidden space.
        # Input x has the shape (batch_size, num_metanodes, hidden_dim)
        # Output embedding has the shape (batch_size, graph_embedding_dim)

        x = x.reshape(-1, self.num_metanodes * self.nodes_hidden_dim)
        graph_embedding = self.mlp(x)
        return graph_embedding

class GraphEmbed(nn.Module):
    def __init__(self,  graph_embedding_dim, num_metanodes, nodes_hidden_dim):
        super(GraphEmbed, self).__init__()

        # Setting from the paper
        self.graph_hidden_size = graph_embedding_dim

        # Embed graphs
        self.node_gating = nn.Sequential(
            nn.Linear(nodes_hidden_dim, 1), nn.Sigmoid()
        )
        self.node_to_graph = nn.Linear(nodes_hidden_dim, self.graph_hidden_size)

    def forward(self, x):
        # Input x has the shape (batch_size, num_metanodes, hidden_dim)
        # Output embedding has the shape (batch_size, graph_embedding_dim)

        x = self.node_gating(x) * self.node_to_graph(x)
        return x.sum(1)


class Hierarchical_GraphEmbed(nn.Module):
    def __init__(self,  graph_embedding_dim, nodes_hidden_dim, dsn_type_start_indices):
        super(Hierarchical_GraphEmbed, self).__init__()

        # Setting from the paper
        self.graph_hidden_size = graph_embedding_dim
        self.nodes_hidden_dim = nodes_hidden_dim
        self.dsn_type_start_indices = dsn_type_start_indices

        # gate nodes
        self.node_gating = nn.Sequential(
            nn.Linear(nodes_hidden_dim, 1), nn.Sigmoid()
        )

        # gate subgraphs
        self.subgraph_gating = nn.Sequential(
            nn.Linear(self.graph_hidden_size * 2, 1), nn.Sigmoid()
        )

        # embedding layers
        self.node_to_subgraph = nn.Linear(nodes_hidden_dim, self.graph_hidden_size)
        self.subgraph_to_graph = nn.Linear(self.graph_hidden_size * 2, self.graph_hidden_size)



    def forward(self, x):
        # Input x has the shape (batch_size, num_metanodes, hidden_dim)
        # Output embedding has the shape (batch_size, graph_embedding_dim)

        # First, apply the node gating mechanism so that the nodes can decide how much they want to contribute to the graph embedding
        x = self.node_gating(x) * self.node_to_subgraph(x)

        # Now, consider subgraphs for the different types of DSNs, combine max and mean pooling to get embeddings for each type of DSN
        pooled_embeddings = torch.empty(x.shape[0], len(self.dsn_type_start_indices), self.graph_hidden_size*2, device=x.device)
        for i in range(len(self.dsn_type_start_indices)):
            start_index = self.dsn_type_start_indices[i]
            end_index = self.dsn_type_start_indices[i + 1] if i + 1 < len(self.dsn_type_start_indices) else x.shape[1]
            subgraph = x[:, start_index:end_index, :]
            max_pooled = torch.max(subgraph, dim=1)[0]
            mean_pooled = torch.mean(subgraph, dim=1)
            mixed_pooled = torch.cat((max_pooled, mean_pooled), dim=-1)
            pooled_embeddings[:, i, :] = mixed_pooled
        # Now, transform the subgraph embeddings, then apply the subgraph gating mechanism
        pooled_embeddings = self.subgraph_gating(pooled_embeddings) * self.subgraph_to_graph(pooled_embeddings)

        # Finally, do mixed max+average pooling to get the final graph embedding
        avg_graph_embedding = torch.mean(pooled_embeddings, dim=1)
        max_graph_embedding = torch.max(pooled_embeddings, dim=1)[0]
        graph_embedding = torch.cat((avg_graph_embedding, max_graph_embedding), dim=-1)
        graph_embedding = self.subgraph_to_graph(graph_embedding)

        
        return graph_embedding


class InitNodeModule(nn.Module):

    def __init__(self, num_metanodes, nodes_hidden_dim, time_information_dim, init_hidden_dim = 4):
        super().__init__()
        self.num_metanodes = num_metanodes
        self.nodes_hidden_dim = nodes_hidden_dim
        self.time_information_dim = time_information_dim

        self.init_hidden_dim = min(init_hidden_dim, nodes_hidden_dim)

        self.init_meta_node_linear1 = nn.Linear(self.time_information_dim, self.num_metanodes * self.init_hidden_dim)
        self.init_meta_node_linear2 = nn.Linear(self.init_hidden_dim, self.nodes_hidden_dim)

    def forward(self, x):
        # x is the temporal context
        # x shape is (batch_size, 2)
        # Output shape is (batch_size, num_metanodes, nodes_hidden_dim)
        x = self.init_meta_node_linear1(x)
        x = F.relu(x)
        x = x.reshape(-1, self.num_metanodes, self.init_hidden_dim)
        x = self.init_meta_node_linear2(x)
        x = torch.tanh(x)
        return x

class GNN_Module(nn.Module):
    def __init__(self, node_hidden_dim,
                 device='cpu', conv_num_layers=3):
        super(GNN_Module, self).__init__()
        self.node_feature_dim = node_hidden_dim      
        self.conv_layers_num = conv_num_layers
        self.conv_hidden_dim = node_hidden_dim

        self.convs = nn.ModuleList()
        for _ in range(self.conv_layers_num):
            self.convs.append(pyg_nn.GCNConv(self.conv_hidden_dim, self.conv_hidden_dim, add_self_loops=False, normalize=False))
        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(self.conv_hidden_dim) for _ in range(self.conv_layers_num)
        )
        
        # initialize weights using xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        param.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)

    def forward(self, X, adj_mat):
        X_gnn_list = []
        for batch_i in range(X.shape[0]):
            edge_indices, edge_attrs = pyg_utils.dense_to_sparse(adj_mat[batch_i])
            X_gnn = X[batch_i]
            for stack_i in range(self.conv_layers_num):
                X_res = X_gnn # Store the current state for the residual connection
                X_gnn = self.convs[stack_i](X_gnn, edge_indices, edge_attrs)
                X_gnn = F.relu(self.layer_norms[stack_i](X_gnn + X_res)) # Add the residual connection
            X_gnn_list.append(X_gnn)
        X_gnn = torch.stack(X_gnn_list, dim=0)
        return X_gnn




class TrafficReconGNN(BaseModel):
    def __init__(self, nodes_hidden_dim, edge_embedding_dim, graph_embedding_dim, observation_hidden_dim, query_horizon, num_attention_heads,
                 neighborhood_shapes,
                    num_temporal_metanodes, num_aq_metanodes, num_weather_metanodes, metadata_dict,  num_gru_layers=2,
                    assignment_threshold=0.015, spatial_assignment_threshold = .8, alpha = .5, detach_query_embedding = False, detach_loss_embedding = False, 
                    use_spatial_dsn = True, use_temporal_dsn = True, use_semantic_dsn = True, use_weather_dsn = True, use_aq_dsn = True,
                     use_streest_as_dsn = True, use_gcn_layer = True, aggregation_methods = 'attbigru',
                     dropout = 0.0, dropout_dsns = False, global_embedding = False, num_dynamical_dsns = 0, monitor_obs_distribution = False,
                     use_message_passing = True, context_in_embedding = True, seq_length = 12):
        super(TrafficReconGNN, self).__init__(0, 0, 1)
        self.nodes_hidden_dim = nodes_hidden_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.observation_hidden_dim = observation_hidden_dim
        self.query_horizon = query_horizon
        self.num_traffic_features = 1
        self.global_embedding = global_embedding
        self.monitor_obs_distribution = monitor_obs_distribution
        self.context_in_embedding = context_in_embedding

        self.metadata_feature_offset = 3

        
        if use_spatial_dsn:
            self.coordinate_inidces = np.arange(self.metadata_feature_offset + metadata_dict['location']['feature_offset'], self.metadata_feature_offset + metadata_dict['location']['feature_offset'] + metadata_dict['location']['feature_num'])
            self.num_streets = metadata_dict['road']['feature_num'] if use_streest_as_dsn else 0
            self.road_feature_indices = np.arange(self.metadata_feature_offset + metadata_dict['road']['feature_offset'], self.metadata_feature_offset + metadata_dict['road']['feature_offset'] + metadata_dict['road']['feature_num'])

            self.num_regions = metadata_dict['region']['feature_num']
            self.region_feature_indices = np.arange(self.metadata_feature_offset + metadata_dict['region']['feature_offset'], self.metadata_feature_offset + metadata_dict['region']['feature_offset'] + metadata_dict['region']['feature_num'])
        
        if use_semantic_dsn:
            self.num_directions = metadata_dict['direction']['feature_num']
            self.num_lanes = metadata_dict['lanes']['max_lanes']
            self.lane_feature_index = metadata_dict['lanes']['feature_offset']
            self.direction_feature_indices = np.arange(self.metadata_feature_offset + metadata_dict['direction']['feature_offset'], self.metadata_feature_offset + metadata_dict['direction']['feature_offset'] + self.num_directions)
        
        self.detach_query_embedding = detach_query_embedding
        self.detach_loss_embedding = detach_loss_embedding

        self.assignment_threshold = assignment_threshold
        self.spatial_assignment_threshold = spatial_assignment_threshold
        self.use_spatial_dsn = use_spatial_dsn
        self.use_temporal_dsn = use_temporal_dsn
        self.use_semantic_dsn = use_semantic_dsn
        self.use_weather_dsn = use_weather_dsn
        self.use_aq_dsn = use_aq_dsn
        self.use_gc_layer = use_gcn_layer
        self.aggregation_methods = aggregation_methods
        self.dropout_dsns = dropout_dsns
        self.dropout = dropout
        self.num_dynamical_dsns = num_dynamical_dsns
        self.num_attention_heads = num_attention_heads
        self.use_message_passing = use_message_passing
        self.seq_length = seq_length
        self.metadata_dict = metadata_dict
                
        # # Define the meta-nodes
        # # Define the spatial metanodes based on the neighborhood_shapes
        # self.regional_metanodes = self.allocate_spatial_metanodes(neighborhood_shapes)
        # self.street_metanodes = self.allocate_spatial_metanodes(num_streets)

        # self.spatial_metanodes = torch.cat((self.regional_metanodes, self.street_metanodes), dim=0)

        # # Define the temporal metanodes
        # self.temporal_metanodes = self.allocate_temporal_metanodes(num_temporal_metanodes)

        # # Define the air quality metanodes
        # self.aq_metanodes = self.allocate_environmental_metanodes(num_aq_metanodes)
        # self.weather_metanodes = self.allocate_environmental_metanodes(num_weather_metanodes)

        # self.environmental_metanodes = torch.cat((self.aq_metanodes, self.weather_metanodes), dim=0)

        # # Define the semantical metanodes
        # self.road_type_metanodes = self.allocate_semantic_metanodes(num_road_types)
        # self.lane_metanodes = self.allocate_semantic_metanodes(num_lanes)
        # self.speed_metanodes = self.allocate_semantic_metanodes(num_speeds)
        # self.lane_type_metanodes = self.allocate_semantic_metanodes(num_lane_types)

        # self.semantic_metanodes = torch.cat((self.road_type_metanodes, self.lane_metanodes, self.speed_metanodes, self.lane_type_metanodes), dim=0)

        # Taking the neighborhood shapes and creating the reference points for the neighborhoods
        # The neighborhood shapes are an imported geojson file that contains the shapes of the neighborhoods
        # Extract the geometry and take the average of all the coordinates to get the reference point for each neighborhood
        self.neighborhood_centers = torch.from_numpy(np.array([np.mean(neighborhood_shape['geometry']['coordinates'][0], axis = 0) for neighborhood_shape in neighborhood_shapes['features'] if neighborhood_shape['geometry']['type'] == 'Polygon']))


        self.neighborhood_centers = torch.nn.Parameter(self.neighborhood_centers.to(torch.float32), requires_grad=False)

        # Calculate the average distance between the reference points to use as the threshold for the spatial assignment
        # Use the distance function as in the spatial assignment module to calculate the distance between the reference points
        neighborhood_lon = torch.deg2rad(self.neighborhood_centers[:, 0])
        neighborhood_lat = torch.deg2rad(self.neighborhood_centers[:, 1])

        min_coord = np.array(
            [self.metadata_dict['location']['min_lat'], self.metadata_dict['location']['min_lng']]
        ).reshape(1,2)

        self.min_coord = torch.nn.Parameter(torch.from_numpy(min_coord).to(torch.float32), requires_grad=False)

        max_coord = np.array(
            [self.metadata_dict['location']['max_lat'], self.metadata_dict['location']['max_lng']]
        ).reshape(1,2)
        self.max_coord = torch.nn.Parameter(torch.from_numpy(max_coord).to(torch.float32), requires_grad=False)
        
        

        x = (neighborhood_lon.unsqueeze(0) - neighborhood_lon.unsqueeze(1)) * torch.cos((neighborhood_lat.unsqueeze(0) + neighborhood_lat.unsqueeze(1)) / 2)
        y = (neighborhood_lat.unsqueeze(0) - neighborhood_lat.unsqueeze(1))

        distance = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2)) * 6317 # distance in kilometers

        self.neighborhood_mean_distance = torch.mean(distance)
        self.neighborhood_std_distance = torch.std(distance)
        print('Neighborhood mean distance:', self.neighborhood_mean_distance)

        
        # Concat all the metanodes for convenience
        # self.all_metanodes = torch.cat((self.spatial_metanodes, self.temporal_metanodes, self.environmental_metanodes, self.semantic_metanodes), dim=0)
        self.num_spatial_metanodes = self.neighborhood_centers.shape[0] + self.num_regions + self.num_streets if self.use_spatial_dsn else 0
        self.num_temporal_metanodes = num_temporal_metanodes if self.use_temporal_dsn else 0
        self.num_aq_metanodes = num_aq_metanodes if self.use_aq_dsn else 0
        self.num_weather_metanodes = num_weather_metanodes if self.use_weather_dsn else 0
        self.num_semantic_metanodes = self.num_lanes - 1  + self.num_directions if self.use_semantic_dsn else 0

        # Define the tensors for Laplacian matrix
        self.num_metanodes = self.num_spatial_metanodes + self.num_temporal_metanodes + self.num_aq_metanodes + self.num_weather_metanodes + self.num_semantic_metanodes + self.num_dynamical_dsns
        self.metanode_edge_embeddings_source = nn.Parameter(torch.Tensor(self.num_metanodes, edge_embedding_dim).uniform_(-1, 1))
        self.metanode_edge_embeddings_target = nn.Parameter(torch.Tensor(self.num_metanodes, edge_embedding_dim).uniform_(-1, 1))
        # alpha is a trainable parameter for the laplacian matrix (how much multihead attention is important, how much the edge embeddings are important.)
        self.alpha = alpha #nn.Parameter(torch.tensor(0.5), requires_grad=True)

        # Define the temporal meta-node centroids here
        self.temporal_metanode_centroids = nn.Parameter(torch.Tensor(num_temporal_metanodes, self.seq_length, self.num_traffic_features).uniform_(-1, 1))
        # Define transformation layers


        self.total_input_dim = self.metadata_feature_offset + np.sum([metadata_dict[k]['feature_num'] for k in metadata_dict])
        
        def create_observation_embedding_module():
            # TODO: Do you agree with the change to dynamic_feature_indices and static-feature_indices?
            # return ObservationEmbeddingModule(self.num_traffic_features, dynamic_feature_indices=[2, 3, 4, 5, 6, 7, 8, 9, 10], static_feature_indices=[11, 12, 13],
            #                                   nodes_hidden_dim=nodes_hidden_dim, num_gru_layers=num_gru_layers, num_hidden_mlp_layers=2)
            return ObservationEmbeddingModule(self.num_traffic_features, 
                                              dynamic_feature_indices=np.array([1,2]),
                                              static_feature_indices=np.arange(self.metadata_feature_offset, self.total_input_dim),
                                              nodes_hidden_dim=self.nodes_hidden_dim, num_gru_layers=num_gru_layers, num_hidden_mlp_layers=2, observation_hidden_dim= self.observation_hidden_dim,
                                              context_in_embedding = self.context_in_embedding)
        
        def create_observation_aggregation_module():
            if self.aggregation_methods == 'attbigru':
                return BatchAttBiGRU(self.observation_hidden_dim, nodes_hidden_dim, num_gru_layers, attention_hidden_dim=nodes_hidden_dim)
            elif self.aggregation_methods == 'transformer':
                return nn.TransformerEncoder( 
                    encoder_layer= nn.TransformerEncoderLayer(d_model=self.nodes_hidden_dim, nhead=num_attention_heads),
                    num_layers= num_gru_layers
                )
            else:
                return None
        
        if self.global_embedding:
            self.global_obs_embedding = create_observation_embedding_module()

        if self.use_aq_dsn:
            raise NotImplementedError("Air quality DSN is not implemented yet.")
            # self.aq_embedding_aggregation_module = create_observation_aggregation_module()
            # self.aq_observation_embedding = create_observation_embedding_module()
            # self.aq_assignment_module = ClusterAssignmentModule(num_attention_heads, nodes_hidden_dim, considered_features_indices=dataset.get_feature_indices(dataset.aq_feature_names), reconstruction_horizon = self.dataset.reconstruction_horizon)

        
        if self.use_weather_dsn:
            raise NotImplementedError("Weather DSN is not implemented yet.")
            # self.weather_embedding_aggregation_module = create_observation_aggregation_module()
            # self.weather_observation_embedding = create_observation_embedding_module()
            # self.weather_assignment_module = ClusterAssignmentModule(num_attention_heads, nodes_hidden_dim, considered_features_indices=dataset.get_feature_indices(dataset.weather_feature_names), reconstruction_horizon = self.dataset.reconstruction_horizon)


        if self.use_spatial_dsn:
            self.spatial_embedding_aggregation_module = create_observation_aggregation_module()
            self.spatial_observation_embedding = create_observation_embedding_module()

        if self.use_semantic_dsn:
            self.semantic_embedding_aggregation_module = create_observation_aggregation_module()
            self.semantic_observation_embedding = create_observation_embedding_module()

        if self.use_temporal_dsn:
            self.temporal_embedding_aggregation_module = create_observation_aggregation_module()     
            self.temporal_observation_embedding = create_observation_embedding_module()

        if self.num_dynamical_dsns > 0:
            self.dyn_embedding_aggregation_module = create_observation_aggregation_module()
            self.dyn_observation_embedding = create_observation_embedding_module()
            self.dyn_assignment_module = ClusterAssignmentModule(
                num_attention_heads, nodes_hidden_dim, 
                considered_features_indices= np.arange(self.num_traffic_features, self.total_input_dim + 1),  # All features except the first one (which is the traffic value)
            )


        # self.weather_observation_embedding = self.aq_observation_embedding
        # self.spatial_observation_embedding = self.aq_observation_embedding
        # self.semantic_observation_embedding = self.aq_observation_embedding
        # self.temporal_observation_embedding = self.aq_observation_embedding

        # Initialize the weights
        # TODO: Implement this function after knowing the layers of your model
        # self.init_meta_node_linear1 = nn.Linear(dataset.time_information.shape[1], self.nodes_hidden_dim)
        # self.init_meta_node_linear2 = nn.Linear(self.nodes_hidden_dim, self.num_metanodes * self.nodes_hidden_dim)

        self.init_module = InitNodeModule(self.num_metanodes, self.nodes_hidden_dim, 2, init_hidden_dim=4)


        self.meta_node_laplacian_multi_head = nn.MultiheadAttention(self.nodes_hidden_dim, num_attention_heads, batch_first= True)

        # self.graph_embedding_layer = MetanodeGraphEmbedding(graph_embedding_dim, self.num_metanodes, self.nodes_hidden_dim)
        # self.graph_embedding_layer = GraphEmbed(graph_embedding_dim, self.num_metanodes, self.nodes_hidden_dim)
        

        counts_of_dsn_types = []
        if self.use_spatial_dsn:
            counts_of_dsn_types.append(self.num_spatial_metanodes)
        if self.use_temporal_dsn:
            counts_of_dsn_types.append(self.num_temporal_metanodes)
        if self.use_aq_dsn:
            counts_of_dsn_types.append(self.num_aq_metanodes)
        if self.use_weather_dsn:
            counts_of_dsn_types.append(self.num_weather_metanodes)
        if self.use_semantic_dsn:
            counts_of_dsn_types.append(self.num_semantic_metanodes)
        if self.num_dynamical_dsns > 0:
            counts_of_dsn_types.append(self.num_dynamical_dsns)

        starts_of_dsn_types = [sum(counts_of_dsn_types[:i]) for i in range(len(counts_of_dsn_types))]
        self.graph_embedding_layer = Hierarchical_GraphEmbed(graph_embedding_dim, self.nodes_hidden_dim, starts_of_dsn_types)

        self.reconstruction_mlp = nn.Sequential(
            nn.Linear(graph_embedding_dim + 2* nodes_hidden_dim +
                    self.seq_length * (self.total_input_dim - self.num_traffic_features), nodes_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nodes_hidden_dim, nodes_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nodes_hidden_dim, nodes_hidden_dim),
            nn.ReLU(),
            nn.Linear(nodes_hidden_dim, self.query_horizon * self.num_traffic_features)
        )

        self.gnn_block = GNN_Module(self.nodes_hidden_dim, conv_num_layers=3)


        
        self.dsn_dropout = nn.Dropout(dropout)

        self.observation_embed_to_node_embed_layer = nn.Linear(self.observation_hidden_dim, self.nodes_hidden_dim)
        self.multi_head_obs_aggregation = nn.MultiheadAttention(self.nodes_hidden_dim, self.num_attention_heads, batch_first= True, kdim=self.observation_hidden_dim, vdim=self.nodes_hidden_dim)

        self.meanminmax_mlps = nn.Sequential(
            nn.Linear(self.observation_hidden_dim * 3, self.nodes_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.nodes_hidden_dim, self.nodes_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.nodes_hidden_dim, self.nodes_hidden_dim)
        )

        self.zero_spatial_dsn_after_aggregation = False
        self.zero_temporal_dsn_after_aggregation = False
        self.zero_env_dsn_after_aggregation = False
        self.zero_semantic_dsn_after_aggregation = False



        self.reset_assignments_monitor()

    def reset_assignments_monitor(self):

        if not self.monitor_obs_distribution:
            return 

        self.spatial_assignments_monitor = torch.zeros((self.num_spatial_metanodes,))
        self.temporal_assignments_monitor = torch.zeros((self.num_temporal_metanodes,))
        self.aq_assignments_monitor = torch.zeros((self.num_aq_metanodes,))
        self.weather_assignments_monitor = torch.zeros((self.num_weather_metanodes,))
        self.semantic_assignments_monitor = torch.zeros((self.num_semantic_metanodes,))
        self.dyn_assignments_monitor = torch.zeros((self.num_dynamical_dsns,))

    def get_assignments_monitor(self):
        if not self.monitor_obs_distribution:
            return {}
        # return a dictionary of the assignments monitor
        return {
            'spatial': self.spatial_assignments_monitor,
            'temporal': self.temporal_assignments_monitor,
            'aq': self.aq_assignments_monitor,
            'weather': self.weather_assignments_monitor,
            'semantic': self.semantic_assignments_monitor,
            'dyn': self.dyn_assignments_monitor
        }


    def forward(self, source, target):
        b, t, n, f = source.shape

        # source is the input with shape (batch_size, num_timesteps, num_observations, num_features)

        # x is a tuple
        # x[0] is the temporal context (time of the day and day of the week). The shape is (batch_size, 2)
        # x[1] is a list of tensors. The len of the list is the same as batch_size.
        # Each list item in x[1]: tensor of observations in a timestamp. the shape is (#num_timesteps, #num_observations, #num_features)
        # x[2] is the query input. The shape is (batch_size, #num_queries, #num_features)
        temporal_context = source[:, 0, 0, 1:3]

        # forward pass
        # first, initialize the state of the metanodes based on the temporal context
        # metanode_states = self.init_metanode_states(temporal_context).reshape(-1,  self.num_metanodes, self.nodes_hidden_dim)
        metanode_states = self.init_module(temporal_context)

        # Divide the input into spatial, temporal, environmental, and semantic observations
        meta_node_dict = {}

        observations = source
        observation_counts_in_batch = [obs.shape[1] for obs in observations]
        # batch_merged_in_obs_dim = torch.cat(observations, dim=1)
        batch_merged_in_obs_dim = observations.transpose(0,1).reshape(t, b*n, f)

        if self.global_embedding: 
            global_observation_embedding = self.global_obs_embedding(batch_merged_in_obs_dim)

        
        with record_function('TrafficReconGNN.forward.spatial'):
            if self.use_spatial_dsn:
                spatial_metanode_states = metanode_states[:, :self.num_spatial_metanodes]

                if self.zero_spatial_dsn_after_aggregation:
                    spatial_metanode_states = spatial_metanode_states * 0
                else:

                    spatial_obs_embedding, _ = self.spatial_observation_embedding(batch_merged_in_obs_dim) if not self.global_embedding else global_observation_embedding

                    spatial_assignment_scores  = self.spatial_assignment(batch_merged_in_obs_dim, self.neighborhood_centers, self.coordinate_inidces)

                    if self.monitor_obs_distribution:
                        mean_number_of_observations_in_a_metanode_per_batch_spatial = torch.sum(spatial_assignment_scores > self.spatial_assignment_threshold, dim=0).detach().to(self.spatial_assignments_monitor)
                        self.spatial_assignments_monitor = self.spatial_assignments_monitor + mean_number_of_observations_in_a_metanode_per_batch_spatial

                    thresholded_spatial_assignment_scores = torch.where(spatial_assignment_scores > self.spatial_assignment_threshold, spatial_assignment_scores, torch.zeros_like(spatial_assignment_scores))  

                    weighted_embedded_spatial_observations = torch.einsum('qh,qm->mqh', spatial_obs_embedding, thresholded_spatial_assignment_scores) # q = b + o as it is the combined dimension of observation and batch 

                    spatial_metanode_states = spatial_metanode_states + self.aggregate_observations_per_metanode(weighted_embedded_spatial_observations, thresholded_spatial_assignment_scores, observation_counts_in_batch, self.spatial_embedding_aggregation_module, spatial_metanode_states)

                meta_node_dict['spatial'] = spatial_metanode_states
        with record_function('TrafficReconGNN.forward.temporal'):
            if self.use_temporal_dsn:
                temporal_metanode_states = metanode_states[:, self.num_spatial_metanodes:self.num_spatial_metanodes+self.num_temporal_metanodes]
                
                if self.zero_temporal_dsn_after_aggregation:
                    temporal_metanode_states = temporal_metanode_states * 0
                else:
                    temporal_obs_embedding, traffic_embeddings = self.temporal_observation_embedding(batch_merged_in_obs_dim) if not self.global_embedding else global_observation_embedding

                    temporal_assignment_scores = self.temporal_assignment(traffic_embeddings)
                    assignment_threshold = torch.kthvalue(temporal_assignment_scores, int((1 - self.assignment_threshold) * temporal_assignment_scores.shape[0]), dim=0).values
                    
                    if self.monitor_obs_distribution:
                        mean_number_of_observations_in_a_metanode_per_batch_temporal = torch.sum(temporal_assignment_scores > assignment_threshold, dim=0).detach().to(self.spatial_assignments_monitor)
                        self.temporal_assignments_monitor = self.temporal_assignments_monitor + mean_number_of_observations_in_a_metanode_per_batch_temporal

                    thresholded_temporal_assignment_scores = torch.where(temporal_assignment_scores > assignment_threshold, temporal_assignment_scores, torch.zeros_like(temporal_assignment_scores))

                    weighted_embedded_temporal_observations = torch.einsum('qh,qm->mqh', temporal_obs_embedding, thresholded_temporal_assignment_scores) # q = b + o as it is the combined dimension of observation and batch
                    
                    temporal_metanode_states = temporal_metanode_states + self.aggregate_observations_per_metanode(weighted_embedded_temporal_observations, thresholded_temporal_assignment_scores, observation_counts_in_batch, self.temporal_embedding_aggregation_module, temporal_metanode_states)
                
                meta_node_dict['temporal'] = temporal_metanode_states
        # environmental_metanode_states = metanode_states[:, self.num_spatial_metanodes+self.num_temporal_metanodes:self.num_spatial_metanodes+self.num_temporal_metanodes+self.num_environmental_metanodes]
        if self.use_aq_dsn:
            aq_metanode_states = metanode_states[:, self.num_spatial_metanodes+self.num_temporal_metanodes:self.num_spatial_metanodes+self.num_temporal_metanodes+self.num_aq_metanodes]
            
            if self.zero_env_dsn_after_aggregation:
                aq_metanode_states = aq_metanode_states * 0
            else:
                aq_obs_embedding, _ = self.aq_observation_embedding(batch_merged_in_obs_dim) if not self.global_embedding else global_observation_embedding

                aq_assignment_scores_per_batch = []
                for batch_i, obs_per_batch in enumerate(observations):
                    aq_assignment_scores = self.aq_assignment_module(aq_metanode_states[batch_i], obs_per_batch)
                    # aq_assignment_scores = self.dataset.get_aq_cluster_series(obs_per_batch[..., self.dataset.get_feature_indices(self.dataset.aq_feature_names)].transpose(0,1).reshape(-1, len(self.dataset.aq_feature_names) * self.dataset.reconstruction_horizon))
                    aq_assignment_scores_per_batch.append(aq_assignment_scores)

                aq_assignment_scores = torch.cat(aq_assignment_scores_per_batch, dim=0)
                assignment_threshold = torch.kthvalue(aq_assignment_scores, int((1 - self.assignment_threshold) * aq_assignment_scores.shape[0]), dim=0).values

                if self.monitor_obs_distribution:
                    mean_number_of_observations_in_a_metanode_per_batch_aq = torch.sum(aq_assignment_scores > assignment_threshold, dim=0).detach().to(self.spatial_assignments_monitor)
                    self.aq_assignments_monitor = self.aq_assignments_monitor + mean_number_of_observations_in_a_metanode_per_batch_aq

                thresholded_aq_assignment_scores = torch.where(aq_assignment_scores > assignment_threshold, aq_assignment_scores, torch.zeros_like(aq_assignment_scores))

                weighted_embedded_aq_observations = torch.einsum('qh,qm->mqh', aq_obs_embedding, thresholded_aq_assignment_scores) # q = b + o as it is the combined dimension of observation and batch

                aq_metanode_states = aq_metanode_states + self.aggregate_observations_per_metanode(weighted_embedded_aq_observations, thresholded_aq_assignment_scores, observation_counts_in_batch, self.aq_embedding_aggregation_module, aq_metanode_states)

            meta_node_dict['aq'] = aq_metanode_states
        if self.use_weather_dsn:
            weather_metanode_states = metanode_states[:, self.num_spatial_metanodes+self.num_temporal_metanodes+self.num_aq_metanodes:self.num_spatial_metanodes+self.num_temporal_metanodes+self.num_aq_metanodes+self.num_weather_metanodes]
            if self.zero_env_dsn_after_aggregation:
                weather_metanode_states = weather_metanode_states * 0
            else:
                weather_obs_embedding, _ = self.weather_observation_embedding(batch_merged_in_obs_dim) if not self.global_embedding else global_observation_embedding

                weather_assignment_scores_per_batch = []
                for batch_i, obs_per_batch in enumerate(observations):
                    weather_assignment_scores = self.weather_assignment_module(weather_metanode_states[batch_i], obs_per_batch)

                    # weather_assignment_scores = self.dataset.get_weather_cluster_series(obs_per_batch[..., self.dataset.get_feature_indices(self.dataset.weather_feature_names)].transpose(0,1).reshape(-1, len(self.dataset.weather_feature_names) * self.dataset.reconstruction_horizon))
                    weather_assignment_scores_per_batch.append(weather_assignment_scores)

                weather_assignment_scores = torch.cat(weather_assignment_scores_per_batch, dim=0)
                assignment_threshold = torch.kthvalue(weather_assignment_scores, int((1 - self.assignment_threshold) * weather_assignment_scores.shape[0]), dim=0).values

                if self.monitor_obs_distribution:
                    mean_number_of_observations_in_a_metanode_per_batch_weather = torch.sum(weather_assignment_scores > assignment_threshold, dim=0).detach().to(self.spatial_assignments_monitor)
                    self.weather_assignments_monitor = self.weather_assignments_monitor + mean_number_of_observations_in_a_metanode_per_batch_weather


                thresholded_weather_assignment_scores = torch.where(weather_assignment_scores > assignment_threshold, weather_assignment_scores, torch.zeros_like(weather_assignment_scores))

                weighted_embedded_weather_observations = torch.einsum('qh,qm->mqh', weather_obs_embedding, thresholded_weather_assignment_scores) # q = b + o as it is the combined dimension of observation and batch

                weather_metanode_states = weather_metanode_states + self.aggregate_observations_per_metanode(weighted_embedded_weather_observations, thresholded_weather_assignment_scores, observation_counts_in_batch, self.weather_embedding_aggregation_module, weather_metanode_states)

            meta_node_dict['weather'] = weather_metanode_states
        with record_function('TrafficReconGNN.forward.semantic'):
            if self.use_semantic_dsn:    
                semantic_metanode_states = metanode_states[:, self.num_spatial_metanodes+self.num_temporal_metanodes+self.num_aq_metanodes+self.num_weather_metanodes:self.num_spatial_metanodes+self.num_temporal_metanodes+self.num_aq_metanodes+self.num_weather_metanodes + self.num_semantic_metanodes]
                
                if self.zero_semantic_dsn_after_aggregation:
                    semantic_metanode_states = semantic_metanode_states * 0
                else:
                    semantic_obs_embedding, _ = self.semantic_observation_embedding(batch_merged_in_obs_dim) if not self.global_embedding else global_observation_embedding

                    semantic_assignment_scores = self.assign_observation_to_semantic_metanodes(batch_merged_in_obs_dim)

                    if self.monitor_obs_distribution:
                        mean_number_of_observations_in_a_metanode_per_batch_semantic = torch.sum(semantic_assignment_scores > self.assignment_threshold, dim=0).detach().to(self.spatial_assignments_monitor)
                        self.semantic_assignments_monitor = self.semantic_assignments_monitor + mean_number_of_observations_in_a_metanode_per_batch_semantic

                    thresholded_semantic_assignment_scores = torch.where(semantic_assignment_scores > self.assignment_threshold, semantic_assignment_scores, torch.zeros_like(semantic_assignment_scores))

                    weighted_embedded_semantic_observations = torch.einsum('qh,qm->mqh', semantic_obs_embedding, thresholded_semantic_assignment_scores) # q = b + o as it is the combined dimension of observation and batch

                    semantic_metanode_states = semantic_metanode_states + self.aggregate_observations_per_metanode(weighted_embedded_semantic_observations, thresholded_semantic_assignment_scores, observation_counts_in_batch, self.semantic_embedding_aggregation_module, semantic_metanode_states)

                    meta_node_dict['semantic'] = semantic_metanode_states
        with record_function('TrafficReconGNN.forward.dyn'):
            if self.num_dynamical_dsns > 0:

                dyn_metanode_states = metanode_states[:, self.num_spatial_metanodes+self.num_temporal_metanodes+self.num_aq_metanodes+self.num_weather_metanodes + self.num_semantic_metanodes : self.num_spatial_metanodes+self.num_temporal_metanodes+self.num_aq_metanodes+self.num_weather_metanodes + self.num_semantic_metanodes + self.num_dynamical_dsns]
                
                
                dyn_obs_embedding, _ = self.dyn_observation_embedding(batch_merged_in_obs_dim) if not self.global_embedding else global_observation_embedding

                # dyn_assignment_scores_per_batch = []
                # for batch_i, obs_per_batch in enumerate(observations):
                #     dyn_assignment_scores = self.dyn_assignment_module(dyn_metanode_states[batch_i], obs_per_batch)
                #     dyn_assignment_scores_per_batch.append(dyn_assignment_scores)

                # dyn_assignment_scores = torch.cat(dyn_assignment_scores_per_batch, dim=0)

                dyn_assignment_scores = torch.zeros(batch_merged_in_obs_dim.shape[1], self.num_dynamical_dsns, device=dyn_obs_embedding.device)

                for i in range(dyn_assignment_scores.shape[0]):
                    dyn_assignment_scores[i, i% self.num_dynamical_dsns] = 1.0

                dyn_assignment_scores = dyn_assignment_scores.detach()

                if self.monitor_obs_distribution:
                    mean_number_of_observations_in_a_metanode_per_batch_dyn = torch.sum(dyn_assignment_scores > self.assignment_threshold, dim=0).detach().to(self.spatial_assignments_monitor)
                    self.dyn_assignments_monitor = self.dyn_assignments_monitor + mean_number_of_observations_in_a_metanode_per_batch_dyn

                thresholded_dyn_assignment_scores = torch.where(dyn_assignment_scores > self.assignment_threshold, dyn_assignment_scores, torch.zeros_like(dyn_assignment_scores))

                weighted_embedded_dyn_observations = torch.einsum('qh,qm->mqh', dyn_obs_embedding, thresholded_dyn_assignment_scores) # q = b + o as it is the combined dimension of observation and batch

                dyn_metanode_states = dyn_metanode_states + self.aggregate_observations_per_metanode(weighted_embedded_dyn_observations, thresholded_dyn_assignment_scores, observation_counts_in_batch, self.dyn_embedding_aggregation_module, dyn_metanode_states)

                meta_node_dict['dyn'] = dyn_metanode_states
       
        # Update the state of the metanodes based on the observations
        
        # environmental_metanode_states = self.update_environmental_metanode_states(environmental_metanode_states, x[1], self.aq_assignment_module, self.aq_observation_embedding,
        #                                           self.aq_embedding_aggregation_module, assignment_threshold=self.assignment_threshold)

        # Now we have the updated meta-node states. Calculate the laplacian matrix
        # pass the metanode states through a multihead attention layer

        # then do a weighted attenton score with the edge embeddings with alpha as weight
        # TODO START: should we remove self-loops for long-term dependencies?
        # base_laplacian = torch.softmax(self.metanode_edge_embeddings.matmul(self.metanode_edge_embeddings.T), dim = -1).unsqueeze(dim = 0)
        
        # Get the long-term dependencies between sensors based on the global node embeddings
        with record_function('TrafficReconGNN.forward.laplacian'):
            scores = torch.matmul(self.metanode_edge_embeddings_source, self.metanode_edge_embeddings_target.T)
            scores = F.relu(scores)
            
            # mask the diagonal elements of the laplacian matrix
            mask = torch.eye(scores.size(0), device=scores.device).bool()
            scores = scores.masked_fill(mask, float('-inf'))
            
            # apply softmax to get the long-term similarity scores
            base_laplacian = F.softmax(scores, dim=-1)
            # TODO: end

            _, attn_weights = self.meta_node_laplacian_multi_head(metanode_states,metanode_states,metanode_states, need_weights=True)
            # multi_head_laplacian = torch.softmax(attn_weights, dim = -1)
            multi_head_laplacian = attn_weights
            laplacian_matrix = self.alpha * base_laplacian + (1 - self.alpha) * multi_head_laplacian
            
            # threshold the laplacian matrix
            laplacian_matrix = self._get_sparse_laplacian(laplacian_matrix, method='percentile', threshold=0.5)

            # Run the GNN message passing (GCN layers)



            all_metanodes_states = torch.cat(list(meta_node_dict.values()), dim=1)
            all_metanodes_states_unconvoluted = all_metanodes_states


            if self.dropout_dsns:
                mask = torch.rand(all_metanodes_states.shape[0], all_metanodes_states.shape[1], device= all_metanodes_states.device) > self.dropout
                mask_values = torch.full_like(mask, fill_value= 1.0 / (1.0 - self.dropout))

                mask = mask.float() * mask_values
                all_metanodes_states = all_metanodes_states * mask.unsqueeze(-1).expand(-1, -1, all_metanodes_states.shape[-1])

            elif self.dropout > 0.0:
                all_metanodes_states = self.dsn_dropout(all_metanodes_states)

            if self.use_gc_layer:
                all_metanodes_states_convoluted = self.gnn_block(all_metanodes_states, laplacian_matrix)
            else:
                all_metanodes_states_convoluted = all_metanodes_states

        with record_function('TrafficReconGNN.forward.reconstruction'):
            
            # Embed the metanode states into a graph-level embedding
            graph_embedding = self.graph_embedding_layer(all_metanodes_states_convoluted)



            reconstructed_query_nodes, query_assignment = self.reconstruct_traffic_features(observations, all_metanodes_states_convoluted, graph_embedding, meta_node_dict, all_metanodes_states_unconvoluted)
        
        return reconstructed_query_nodes
    
    def aggregate_observations_per_metanode(self, 
                                            weighted_embedded_observations, thresholded_assignment_scores, 
                                            observation_counts_in_batch, embedding_aggregation_module, meta_node_states ):
        """
        ### Parameters:

        - meta_node_states: torch.Tensor with shape (batch_size, #num_metanodes, #nodes_hidden_dim)
        - weighted_embedded_observations: torch.Tensor with shape (#num_metanodes, batch_size * #num_observations, #nodes_hidden_dim)
        """


        list_of_weighted_batches = torch.split(weighted_embedded_observations, observation_counts_in_batch, dim=1)
        list_of_assignments = torch.split(thresholded_assignment_scores, observation_counts_in_batch, dim=0)

        if self.aggregation_methods != 'multihead':
            updated_metanode_states_per_batch = []
            for batch_i, (weighted_embedded_observations_per_batch, thresholded_assignment_scores_per_batch) in enumerate(zip(list_of_weighted_batches, list_of_assignments)):

                delta_metanode_states = self.aggregation_observation_embedding(weighted_embedded_observations_per_batch, thresholded_assignment_scores_per_batch, embedding_aggregation_module)
                updated_metanode_states_per_batch.append(delta_metanode_states)

            return torch.stack(updated_metanode_states_per_batch, dim=0)
    
        else:
            return self.batched_multihead_metanodes_observations(meta_node_states, list_of_weighted_batches, observation_counts_in_batch, [loa == 0.0 for loa in list_of_assignments])


    def batched_multihead_metanodes_observations(self, metanode_states, batch_list_of_observations, observations_per_batch, list_of_thresholded_assignments):
        """
        ### Parameters:
            - metanode_states: torch.Tensor with shape (batch_size, #num_metanodes, #nodes_hidden_dim)
            - batch_list_of_observations: list of torch.Tensor with shape (#num_metanodes, #num_observations, #nodes_hidden_dim)
            - observations_per_batch: list of integers with the number of observations in each batch
            - list_of_thresholded_assignments: list of torch.Tensor with shape (#num_observations, #num_metanodes) with 1.0 or True for that this observation i was thresholded for this metanode j

        ### Returns:
            - updated_metanode_states: torch.Tensor with shape (batch_size, #num_metanodes, #nodes_hidden_dim) 
        """
        max_obs = max(observations_per_batch)

        # pad the weighted observations to the same length
        padded_list_of_weighted_obs = [F.pad(batch, (0, 0, 0, max_obs - batch.shape[1], 0, 0)) for batch in batch_list_of_observations]
        padded_list_of_thresholded_assignments = [F.pad(lota, (0,0,0, max_obs - lota.shape[0]), value= 1.0) for lota in list_of_thresholded_assignments]


        weighted_obs_joined_dim_batch_meta = torch.cat(padded_list_of_weighted_obs, dim=0) # shape: (#num_metanodes * batch_size, #num_observations, #nodes_hidden_dim)
        thresholded_mask = torch.cat(padded_list_of_thresholded_assignments, dim=1).transpose(0,1).unsqueeze(-2) # shape: (#num_metanodes * batch_size, 1, #num_observations)
        query_indices_with_at_least_one_observation = (~thresholded_mask.bool()).squeeze(1).any(dim=-1).argwhere().squeeze(dim = -1)

        

        if query_indices_with_at_least_one_observation.numel() == 0:
            return torch.zeros_like(metanode_states)

        thresholded_mask = thresholded_mask[query_indices_with_at_least_one_observation]

        thresholded_mask_for_each_head = thresholded_mask.unsqueeze(1).expand(-1, self.num_attention_heads, -1, -1) # shape: (#num_metanodes * batch_size, #num_attention_heads, 1, #num_observations)
        thresholded_mask_for_each_head = thresholded_mask_for_each_head.reshape(-1, 1, max_obs) # shape: (#num_metanodes * batch_size * #num_attention_heads, 1, #num_observations)

        # merge the two first dimensions of meta node states
        query = metanode_states.reshape(-1, self.nodes_hidden_dim).unsqueeze(1)

        # In the case that a DSN has no observations, the thresholded mask will be all ones forbidding to put any attention to an observations which ceates NANs in the attention weights as the softmax is not defined. In this case also the output updated_metanode_states will be NaNs. Therefore we cut the q,k,v tensors.
        #take only queries with at least one observation
        query_cutted = query[query_indices_with_at_least_one_observation]
        weighted_obs_joined_dim_batch_meta_cutted = weighted_obs_joined_dim_batch_meta[query_indices_with_at_least_one_observation]
        keys = weighted_obs_joined_dim_batch_meta_cutted

        values = self.observation_embed_to_node_embed_layer(weighted_obs_joined_dim_batch_meta_cutted)

        # apply multihead attention
        multihead_attention_output, _ = self.multi_head_obs_aggregation(query_cutted, keys, values, need_weights=False, attn_mask = thresholded_mask_for_each_head.bool())


        updated_metanode_states = torch.zeros_like(query)
        updated_metanode_states[query_indices_with_at_least_one_observation] = multihead_attention_output

        return updated_metanode_states.reshape(metanode_states.shape)




    def temporal_assignment(self, traffic_obs_embedding):
        temporal_centroid_embeddings = self.temporal_observation_embedding.embed_traffic_values(self.temporal_metanode_centroids.permute(1, 0, 2))
        # calculate the assignment scores from observations to temporal metanodes based on the cosine similarity
        # implementing a pairwise cosine similarity function between traffic embeddings and temporal centroids embeddings
        # traffic_norm = traffic_obs_embedding / traffic_obs_embedding.norm(dim=1)[:, None]
        # temporal_centroids_norm = temporal_centroid_embeddings / temporal_centroid_embeddings.norm(dim=1)[:, None]
        # assignment_scores = torch.mm(traffic_norm, temporal_centroids_norm.transpose(0,1))

        distance = torch.cdist(traffic_obs_embedding, temporal_centroid_embeddings, p=2)

        assignment_scores = torch.exp(-distance)

        return assignment_scores
    
    def spatial_assignment(self, batch_merged_in_obs_dim, neighborhood_reference_points, coordinate_feature_indices):
        """
        ### Parameters:
            - batch_merged_in_obs_dim: torch.Tensor with shape (batch_size * #num_observations, #num_features)
            

        ### Returns:
            - assignment_scores: torch.Tensor with shape (batch_size * #num_observations, #spatial_metanodes)
        """
        assignment_scores = self.assign_observation_to_neighborhoods(batch_merged_in_obs_dim, neighborhood_reference_points, coordinate_feature_indices)

        assignment_scores_street = batch_merged_in_obs_dim[0, : , self.road_feature_indices]
        assignment_scores = torch.cat((assignment_scores, assignment_scores_street), dim=1)


        assignment_scores_region = batch_merged_in_obs_dim[0, : , self.region_feature_indices]
        assignment_scores = torch.cat((assignment_scores, assignment_scores_region), dim=1)


        return assignment_scores
        


    def reconstruct_traffic_features(self, query_input, all_metanodes_states, graph_embedding, meta_node_dict, all_metanodes_states_unconvoluted):
        # For each query observation, calculate an assignment score to each metanode
        # Use the embedding observation module from each metanode cateogry (sematical, spatial, temporal, environmental) to create the query node embedding based on the dynamical and static features but withput traffic values.
        # Use those assignment embeddings to calculate the cosine similarity to the meta node embeddings.
        # Take the weights average of the meta node embeddings to get the query node embedding.
        query_and_graph_embeddings_per_batch = []
        assignment_scores_batch = []
        query_nodes_per_batch = [query_batch.shape[1] for query_batch in query_input]
        for batch_i, query_batch, metanode_batch, graph_batch, metanode_unconv_batch in zip(range(len(query_input)),query_input, all_metanodes_states, graph_embedding, all_metanodes_states_unconvoluted):



            # Calculating pairwise cosine similarity between query embeddings and metanode embeddings
            # The shape of assignment_scores is (#batch_size, #num_queries, #num_metanodes) and the query embedding is (#batch_size, #num_queries, self.nodes_hidden_dim) and meta node embedding is (#num_metanodes, self.nodes_hidden_dim)

            def similarity_query_embedding_metanode_state(query_embedding, metanode_states):
                q_norm = query_embedding / (query_embedding.norm(dim=1)[:, None] + 1e-9)
                m_norm = metanode_states / (metanode_states.norm(dim=1)[:, None] + 1e-9)
                res = torch.mm(q_norm, m_norm.transpose(0,1))

                return res

            assignment_dsns = []
            if self.global_embedding:
                query_embedding_global = self.global_obs_embedding.query_embedding(query_batch)

            for meta_node_type_key in meta_node_dict:
                if 'aq' == meta_node_type_key:
                    # query_embedding_aq = self.aq_observation_embedding.query_embedding(query_batch) if not self.global_embedding else query_embedding_global
                    
                    assignment_scores_aq = self.aq_assignment_module(meta_node_dict['aq'][batch_i], query_batch)
                    # assignment_scores_aq = self.dataset.get_aq_cluster_series(query_batch[..., self.dataset.get_feature_indices(self.dataset.aq_feature_names)].transpose(0,1).reshape(-1, len(self.dataset.aq_feature_names) * self.dataset.reconstruction_horizon))

                    # thresholded_aq_assignment_scores = torch.where(assignment_scores_aq > self.assignment_threshold, assignment_scores_aq, torch.zeros_like(assignment_scores_aq))
                    # move the assignment threshold to the 15% highest per cent of the assignment scores
                    assignment_threshold = torch.kthvalue(assignment_scores_aq, int((1 - self.assignment_threshold) * assignment_scores_aq.shape[0]), dim=0).values
                    thresholded_aq_assignment_scores = torch.where(assignment_scores_aq > assignment_threshold, assignment_scores_aq, torch.zeros_like(assignment_scores_aq))


                    assignment_dsns.append(thresholded_aq_assignment_scores)

                if 'weather' == meta_node_type_key:
                    # query_embedding_weather = self.weather_observation_embedding.query_embedding(query_batch) if not self.global_embedding else query_embedding_global
                    assignment_scores_weather = self.weather_assignment_module(meta_node_dict['weather'][batch_i], query_batch)
                    # assignment_scores_weather = self.dataset.get_weather_cluster_series(query_batch[..., self.dataset.get_feature_indices(self.dataset.weather_feature_names)].transpose(0,1).reshape(-1, len(self.dataset.weather_feature_names) * self.dataset.reconstruction_horizon))
                    assignment_threshold = torch.kthvalue(assignment_scores_weather, int((1 - self.assignment_threshold) * assignment_scores_weather.shape[0]), dim=0).values
                    thresholded_weather_assignment_scores = torch.where(assignment_scores_weather > assignment_threshold, assignment_scores_weather, torch.zeros_like(assignment_scores_weather))
                    assignment_dsns.append(thresholded_weather_assignment_scores)

                if 'spatial' == meta_node_type_key:

                    spatial_assignment_scores  = self.spatial_assignment(query_batch, self.neighborhood_centers, self.coordinate_inidces)
                    thresholded_spatial_assignment_scores = torch.where(spatial_assignment_scores > self.spatial_assignment_threshold, spatial_assignment_scores, torch.zeros_like(spatial_assignment_scores))  
                    assignment_dsns.append(thresholded_spatial_assignment_scores)
                    
                if 'semantic' == meta_node_type_key:

                    semantic_assignment_scores = self.assign_observation_to_semantic_metanodes(query_batch)
                    assignment_dsns.append(semantic_assignment_scores)

                if 'temporal' == meta_node_type_key:
                    # query_embedding_temporal = self.temporal_observation_embedding.query_embedding(query_batch) if not self.global_embedding else query_embedding_global
                    # assignment_scores_temporal = similarity_query_embedding_metanode_state(query_embedding_temporal, meta_node_dict['temporal'][batch_i])

                    ## As a query for the reconstruction does not contain traffic values it is hard to assign temporal nodes to queries. We still assign them from the observation and use their information in the MPGNN.
                    # assignment_scores_temporal = torch.zeros_like(assignment_scores_temporal)
                    assignment_scores_temporal = torch.zeros(query_batch.shape[1], meta_node_dict['temporal'][batch_i].shape[0], device=query_batch.device)
                    assignment_dsns.append(assignment_scores_temporal)
                if 'dyn' == meta_node_type_key:
                    query_embedding_dyn = self.dyn_observation_embedding.query_embedding(query_batch) if not self.global_embedding else query_embedding_global
                    assignment_scores_dyn = similarity_query_embedding_metanode_state(query_embedding_dyn, meta_node_dict['dyn'][batch_i])
                    assignment_dsns.append(assignment_scores_dyn)


            # Creating the metanode assignment scores by concatenating the assignment scores from each metanode category
            assignment_scores = torch.cat(assignment_dsns, dim=1)

            # Doing a softmax to get the final assignment scores
            assignment_scores_metanode_normalized = torch.softmax(assignment_scores, dim=-2)
            # assignment_scores = torch.softmax(assignment_scores, dim=-1)

            if self.detach_loss_embedding:
                assignment_scores = assignment_scores.detach()
     
            query_embedding = torch.einsum('qm,me->qe', assignment_scores, metanode_batch)
            query_embedding_unconv = torch.einsum('qm,me->qe', assignment_scores, metanode_unconv_batch)

            if self.detach_query_embedding:
                query_embedding = query_embedding.detach()
                query_embedding_unconv = query_embedding_unconv.detach()


            query_for_recons_mlp = query_batch[..., self.num_traffic_features:].transpose(0,1).reshape(query_batch.shape[1], -1)
            if not self.context_in_embedding:
                query_for_recons_mlp = torch.zeros_like(query_for_recons_mlp)

            if self.use_message_passing:
                query_and_graph_embeddings = torch.cat(
                (
                    query_embedding, 
                    query_embedding_unconv,
                    graph_batch.unsqueeze(0).expand(query_input[batch_i].shape[1], -1),
                    query_for_recons_mlp
                ), dim=-1)
            else:
                query_and_graph_embeddings = torch.cat(
                (
                    torch.zeros_like(query_embedding), 
                    query_embedding_unconv,
                    torch.zeros_like(graph_batch.unsqueeze(0).expand(query_input[batch_i].shape[1], -1)),
                    query_for_recons_mlp
                ), dim=-1)
            
            query_and_graph_embeddings_per_batch.append(query_and_graph_embeddings)
            assignment_scores_batch.append(assignment_scores_metanode_normalized)

        all_query_and_graph_embeddings = torch.concat(query_and_graph_embeddings_per_batch, dim=0)
        # assignment_scores = torch.stack(assignment_scores_batch, dim=0)

        # Caluclating the weighted sum of metanodes embeddings to get the query node embedding
        # query_embedding = torch.einsum('bqm,bme->bqe', assignment_scores, all_metanodes_states)

        # Now concatenate the query embedding with the graph-level embedding
        # query_and_graph_embeddings = torch.cat((query_embedding, graph_embedding.unsqueeze(1).expand(-1, x[2].shape[1], -1)), dim=-1)

        # Pass through an MLP to get the reconstructed traffic estimates for the query nodes
        # The shape of the output is (#batch_size, #num_queries, #reconstruction_horizon, #num_features)
        reconstructed_output = self.reconstruction_mlp(all_query_and_graph_embeddings).reshape(-1, self.query_horizon, self.num_traffic_features)
    
        # reconstructed traffic split by the batch size
        reconstructed_output = torch.split(reconstructed_output, query_nodes_per_batch, dim=0)

        # tranpose the reconstructed output to have the shape of (#batch_size, #reconstruction_horizon, #num_queries, #num_features)
        reconstructed_output = [reconstructed_output_i.permute( 1, 0, 2) for reconstructed_output_i in reconstructed_output]

        return reconstructed_output, assignment_scores_batch
    
    def init_metanode_states(self, temporal_context):
        """
        Initialize the state of the metanodes based on the temporal context
        shape is (#batch_size, #num_metanodes, self.nodes_hidden_dim)
        """      
        # TODO: uncomment the following line and remove the next line
        # init_state = self.init_meta_node_linear1(temporal_context)
        init_state = self.init_meta_node_linear1(temporal_context)
        init_state = F.relu(init_state)
        init_state = self.init_meta_node_linear2(init_state)
        init_state = F.relu(init_state)

        return init_state
    

    def aggregation_observation_embedding(self, weighted_embedded_observations, assignment_scores, aggregation_module = None, dim = 1):
        # weighted_embedded_observations is (#num_metanodes, #num_observations, self.nodes_hidden_dim)
        # assignment_scores is (#num_observations, #num_metanodes)
        aggregation_embeddings = []

        if 'attbigru' == self.aggregation_methods:
            aggregation_embeddings.append(aggregation_module(weighted_embedded_observations, assignment_scores))
        else:
            weighted_embedded_observations = self.observation_embed_to_node_embed_layer(weighted_embedded_observations)


        if 'mean' == self.aggregation_methods:
            aggregation_embeddings.append(torch.sum(weighted_embedded_observations, dim = 1) / (torch.any(weighted_embedded_observations != 0.0, dim = -1).sum(dim = 1).unsqueeze(-1)+ 1e-9))
        if 'max' == self.aggregation_methods:
            aggregation_embeddings.append(torch.max(weighted_embedded_observations, dim = dim )[0])
        if 'sum' == self.aggregation_methods:
            aggregation_embeddings.append(torch.sum(weighted_embedded_observations, dim = dim ))
        if 'wmean' == self.aggregation_methods:
            meta_node_weights = torch.sum(assignment_scores, dim=0)

            summed_features = torch.sum(weighted_embedded_observations, dim = dim )

            aggregated_embeddings = torch.zeros_like(summed_features)

            aggregated_embeddings[meta_node_weights > 0] = aggregated_embeddings[meta_node_weights > 0] / meta_node_weights[meta_node_weights > 0][:, None]

            aggregation_embeddings.append(aggregated_embeddings)
        if 'meanstd' == self.aggregation_methods:
            # sample from a normal distirbution with loc 0 and std 1 for each feature

            loc = torch.mean(weighted_embedded_observations, dim = dim )
            scale = torch.std(weighted_embedded_observations, dim = dim )

            normal_weights = torch.distributions.Normal(torch.tensor(0.0, device = loc.device), torch.tensor(1.0, device = loc.device)).sample(loc.shape)

            reparametrized_weights = loc + scale * normal_weights
            aggregation_embeddings.append(reparametrized_weights)
        if 'transformer' == self.aggregation_methods:
            transformer_embeddings = aggregation_module(weighted_embedded_observations)
            transformer_embeddings = torch.mean(transformer_embeddings, dim = dim )
            aggregation_embeddings.append(transformer_embeddings)
        if 'meanminmax' == self.aggregation_methods:
            mean = torch.mean(weighted_embedded_observations, dim = dim )
            min = torch.min(weighted_embedded_observations, dim = dim )[0]
            max = torch.max(weighted_embedded_observations, dim = dim )[0]



            aggregation_embeddings.append(self.meanminmax_mlps(torch.cat([mean, min, max], dim = -1)))
        
        return torch.cat(aggregation_embeddings, dim=-1)
    


    def assign_observation_to_neighborhoods(self, observations, neighborhood_reference_points, coordinate_feature_indices):
        # TODO: Middle of the neighbordhood polygons as the reference points.
        
        # As haversine are somewhat expensive we can use an equirectunglar preojection
        # In small regions like our study regions this should be ok as we save some expensive operations compared to a haversine distance. (http://www.movable-type.co.uk/scripts/latlong.html see equirectangular approximation)


        

        inverse_scaled = observations[0, :, coordinate_feature_indices] * (self.max_coord - self.min_coord) + self.min_coord

        latlon_obs = torch.deg2rad(inverse_scaled)
        lat_obs = latlon_obs[..., 0].unsqueeze(1)
        lon_obs = latlon_obs[..., 1].unsqueeze(1)



        # lon_obs = self.dataset.scaled_coordinates_to_radians(observations[0, :, coordinate_feature_indices])[..., 0].unsqueeze(1)
        # lat_obs = self.dataset.scaled_coordinates_to_radians(observations[0, :, coordinate_feature_indices])[..., 1].unsqueeze(1)

        inverse_scaled = neighborhood_reference_points * (self.max_coord - self.min_coord) + self.min_coord
        neighborhood_reference_points = torch.deg2rad(inverse_scaled)

        lon_neighborhood = neighborhood_reference_points[:, 0].unsqueeze(0)
        lat_neighborhood = neighborhood_reference_points[:, 1].unsqueeze(0)

        x = (lon_neighborhood - lon_obs) * torch.cos((lat_neighborhood + lat_obs) / 2)
        y = (lat_neighborhood - lat_obs)

        distance = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2)) * 6317 # distance in kilometers
        # sigma = self.neighborhood_mean_distance / 2 # similar to the radius of arterial roads along the ramps
        sigma = self.neighborhood_std_distance

        gaussian_distances = torch.exp(-(distance ** 2) / (2 * sigma ** 2))


        return gaussian_distances.to(torch.float32)

    


    def assign_observation_to_semantic_metanodes(self, observations):
        # input observations have the shape (timestamps, num_observations, features)

        # TODO: Make this in a more comprehensive way. Either input the dataset to the model so we can query the columns or compute all before creating a instance of the model.
                

        # lanes
        number_of_lanes = observations[0, :, self.lane_feature_index]
        # shape (num_obs_in_batch, num_lanes)
        assignment_to_lanes = F.one_hot(torch.round(number_of_lanes * (self.num_lanes - 2) ).long(), num_classes = self.num_lanes -1 )

        assignment_to_direction = observations[0, :, self.direction_feature_indices]

        # assignment_scores shape (num_observations, num_metanodes)
        assignment_scores = torch.concat([assignment_to_lanes, assignment_to_direction], dim = -1)

        return assignment_scores


    def update_semantic_metanode_states(self, metanode_states, observations, 
                                        observation_embedding_module: ObservationEmbeddingModule, embedding_aggregation_module: Attentional_BiGRU,
                                        ):
        # Update the state of the semantic metanodes based on the observations
        updated_metanode_states_per_batch = []
        # Assign for each observation the meta node. Look into the semnatical features and assign a 1 or 0 to it
        for batch_i, batch_observation in enumerate(observations):

            assignment_scores = self.assign_observation_to_semantic_metanodes(batch_observation)
        
            # embed the observations shape (num_observations, num_hidden_dim)
            embedded_observation, _ = observation_embedding_module(batch_observation)

            # multiplying the embeddings with the scores shape (num_metanodes, num_observations, hidden_dim)
            weighted_embedded_observations = torch.einsum('om,oe->moe', assignment_scores, embedded_observation)

            # (using only non-zero observations as input to the aggregation) - this should be taken care of in the aggregation module making it more convenient to keep the tensor.
            # using different meta nodes as the batch parameter of the attention module
            # the output shape (num_metanodes, hidden_dim)
            delta_metanode_states = self.aggregation_observation_embedding(weighted_embedded_observations, assignment_scores, embedding_aggregation_module)


            # add the aggregated embeddings to the meta nodes
            updated_metanode_states_per_batch.append(metanode_states[batch_i] + delta_metanode_states)

        return torch.stack(updated_metanode_states_per_batch, dim=0)    


    def _get_sparse_laplacian(self, laplacian_matrix, method='percentile', threshold=0.5):
        """
        Get a sparse laplacian matrix by thresholding the given laplacian matrix.
        
        Args:
            laplacian_matrix (Tensor): A tensor of shape (batch_size, n_sensors, n_sensors) containing the laplacian matrix.
            method (str): The method to use for thresholding (choices: 'percentile' and 'value').
            threshold (float): The threshold value for thresholding.
            
        Returns:
            sparse_laplacian (Tensor): A tensor of shape (batch_size, n_sensors, n_sensors) containing the sparse laplacian matrix.
        """
        if method == 'percentile':
            batch_size, n_sensors, _ = laplacian_matrix.shape
            sparse_laplacian = torch.zeros_like(laplacian_matrix)
            
            for i in range(batch_size):
                threshold_value = torch.kthvalue(laplacian_matrix[i].flatten(), int(n_sensors * n_sensors * threshold)).values
                sparse_laplacian[i] = torch.where(laplacian_matrix[i] > threshold_value, laplacian_matrix[i], torch.zeros_like(laplacian_matrix[i]))
        
        elif method == 'value':
            sparse_laplacian = torch.where(laplacian_matrix > threshold, laplacian_matrix, torch.zeros_like(laplacian_matrix))
        
        return sparse_laplacian 