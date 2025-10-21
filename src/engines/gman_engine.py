import torch
import numpy as np
from tqdm import tqdm
from src.base.engine import BaseEngine
from src.utils.metrics import masked_mape, masked_rmse, compute_all_metrics


class GMAN_Engine(BaseEngine):
    def __init__(self, **args):
        super(GMAN_Engine, self).__init__(**args)

    # Helper: build TE (temporal embeddings) from X
    def build_te(self, X, horizon, T=288):
        """
        Build GMAN temporal embedding tensor.
        X: [B, num_his, N, F]  (F includes tod, dow at indices 1:3)
        Return TE: [B, num_his + horizon, 2] with (dow, tod)
        """
        # B: batch size, H: num historical steps, N: num nodes, F: num features
        B, H, N, F = X.shape
        # Take time-of-day and day-of-week from first node (same for all nodes)
        # We turn these to integers so GMAN's temporal embedding works properly by one-hot encoding
        tod = (X[:, :, 0, 1] * (T - 1) + 0.5).long().clamp(0, T - 1)
        dow = (X[:, :, 0, 2] * 7 + 0.5).long() % 7

        # Predict future time slots
        steps = torch.arange(1, horizon + 1, device=X.device).unsqueeze(0)
        tod_f = (tod[:, -1:].expand(-1, horizon) + steps) % T
        carry = ((tod[:, -1:].expand(-1, horizon) + steps) // T)
        dow_f = (dow[:, -1:].expand(-1, horizon) + carry) % 7

        tod_all = torch.cat([tod, tod_f], dim=1)
        dow_all = torch.cat([dow, dow_f], dim=1)
        return torch.stack([dow_all, tod_all], dim=-1)  # [B, his+horizon, 2]

    def forward(self, X, label, isTrain=False):
        
        X_input = X[..., 0]  # [B, his, N]
        TE = self.build_te(X, horizon=self.model.horizon)

        pred = self.model(X_input, TE)

        return pred.unsqueeze(-1), label, None