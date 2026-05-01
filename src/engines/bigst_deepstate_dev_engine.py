import torch
import numpy as np
from tqdm import tqdm
from src.base.engine import BaseEngine


class BigST_Engine(BaseEngine):
    def __init__(self, **args):
        super(BigST_Engine, self).__init__(**args)


    def forward(self, X, label, isTrain = False, query_node = None):
        pred_dict = self.model(X, label)
        pred = pred_dict['prediction']
        
        if query_node is not None:
            pred = pred[:, :, query_node, :]
            label = label[:, :, query_node, :]

        return pred, label, pred_dict
    

    def loss(self, pred, label, mask_value, loss_container):
        pred_dict = loss_container
        loss = bigst_loss(
            prediction= pred,
            target= label,
            node_vec1= pred_dict['node_vec1'],
            node_vec2= pred_dict['node_vec2'],
            supports= pred_dict['supports'],
            use_spatial= pred_dict['use_spatial'],
            mask_value= mask_value
        )

        mean_source_attention = torch.norm(pred_dict['assignment_scores_source'], p = 1)
        mean_target_attention = torch.norm(pred_dict['assignment_scores_target'], p = 1)

        # normalize the deep_node_vector
        # normalized = torch.nn.functional.normalize(pred_dict['deep_node_vec'], dim=-1)
        # scalar_product = torch.einsum('df,ef->de', normalized,normalized) # to avoid memory leak
        # cosine_similarity = torch.mean(1 - scalar_product) # [1]

        # print("Cosine Similarity of Deep Node Vector: ", cosine_similarity.detach().item())

        # deep state nodes should be spatially coherent

        # return loss #+ 30.0 * cosine_similarity
        return loss +   .001 * mean_source_attention + .001 * mean_target_attention
        

import torch
import numpy as np
from src.utils.metrics import masked_mae

def spatial_loss(node_vec1, node_vec2, supports, edge_indices):
    B = node_vec1.size(0)
    node_vec1 = node_vec1.permute(1, 0, 2, 3) # [N, B, 1, r]
    node_vec2 = node_vec2.permute(1, 0, 2, 3) # [N, B, 1, r]
    
    node_vec1_end, node_vec2_start = node_vec1[edge_indices[:, 0]], node_vec2[edge_indices[:, 1]] # [E, B, 1, r]
    attn1 = torch.einsum("ebhm,ebhm->ebh", node_vec1_end, node_vec2_start) # [E, B, 1]
    attn1 = attn1.permute(1, 0, 2) # [B, E, 1]

    one_matrix = torch.ones([node_vec2.shape[0]]).to(node_vec1.device)
    node_vec2_sum = torch.einsum("nbhm,n->bhm", node_vec2, one_matrix)
    attn_norm = torch.einsum("nbhm,bhm->nbh", node_vec1, node_vec2_sum)
    
    attn2 = attn_norm[edge_indices[:, 0]]  # [E, B, 1]
    attn2 = attn2.permute(1, 0, 2) # [B, E, 1]
    attn_score = attn1 / attn2 # [B, E, 1]
    
    d_norm = supports[0][edge_indices[:, 0], edge_indices[:, 1]]
    d_norm = d_norm.reshape(1, -1, 1).repeat(B, 1, attn_score.shape[-1])
    spatial_loss = torch.mean(attn_score.log() * d_norm)
    
    return spatial_loss

def bigst_loss(prediction, target, node_vec1, node_vec2, supports, use_spatial, mask_value):
    if use_spatial:
        supports = [support.to(prediction.device) for support in supports]
        edge_indices = torch.nonzero(supports[0] > 0)
        s_loss = spatial_loss(node_vec1, node_vec2, supports, edge_indices)
        return masked_mae(prediction, target, mask_value) - 0.3 * s_loss # 源代码：pipline.py line30
    else:
        masked_mae_loss = masked_mae(prediction, target, mask_value)
        
        
        return masked_mae_loss