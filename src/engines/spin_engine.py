import torch
import numpy as np
from src.base.engine import BaseEngine

class SPIN_Engine(BaseEngine):
    def __init__(self, edge_index, **args):
        super(SPIN_Engine, self).__init__(**args)
        self.edge_index = edge_index    

    def forward(self, X, label, isTrain = False):

        # X = X.permute(0, 2,1,3)  # (B, N, T, F)
        # label = label.permute(0,2,1,3)

        _, n_node, _, n_features = X.shape


        # node_embeddings = torch.arange(n_node, device=X.device).int()

        assert n_features >= 3, 'The input feature should be at least 3, including traffic feature and time feature'


        X_traffic = X[..., [0]]
        X_time = X[:, :, 0, 1:3]

        if n_features > 3:
            X_other = X[..., 3:]
            X_traffic = torch.cat([X_traffic, X_other], dim=-1)
        
        # time and week are global 
        # need the shape B,T,F



        pred = self.model(
            x = X_traffic,
            u = X_time, # not yet transformed in cos and sin
            mask = self.current_x_mask.bool(),  # invert because here True means missing
            edge_index = self.edge_index,
        )

        predictions, imputations = pred

        # requires (B, N, T, F)
  

        # pred = pred.permute(0, 2, 1, 3)
        # label = label.permute(0, 2, 1, 3)
        
        return predictions, label, None
