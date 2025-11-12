import torch
import numpy as np
from src.base.engine import BaseEngine

class OPCR_Engine(BaseEngine):
    def __init__(self, **args):
        super(OPCR_Engine, self).__init__(**args)


    def forward(self, X, label, isTrain = False):

        X = X.permute(0, 2, 1, 3)  # (B, N, T, F)
        label = label.permute(0, 2, 1, 3)

        _, n_node, _, n_features = X.shape


        node_embeddings = torch.arange(n_node, device=X.device).int()

        assert n_features >= 3, 'The input feature should be at least 3, including traffic feature and time feature'


        X_traffic = X[..., [0]]
        X_time = X[:, 0, :, 1:3]

        if n_features > 3:
            X_other = X[..., 3:]
            X_traffic = torch.cat([X_traffic, X_other], dim=-1)
        
        # time and week are global 
        # need the shape B,T,F



        pred = self.model(
            node_embed = node_embeddings,
            x = X_traffic,
            ex = X_time, # not yet transformed in cos and sin
            mask = ~ self.current_x_mask.permute(0, 2, 1, 3).bool(),  # (B, N, T, F) invert because here True means missing
        )

        # requires (B, N, T, F)
  
        pred = pred.permute(0, 2, 1, 3)
        label = label.permute(0, 2, 1, 3)
        
        return pred, label, None
