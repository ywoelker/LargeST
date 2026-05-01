import torch
import numpy as np
from tqdm import tqdm
from src.base.engine import BaseEngine
from src.utils.metrics import masked_mape, masked_rmse
from src.utils.metrics import compute_all_metrics

from torch.profiler import profile, record_function, ProfilerActivity


class DeepStateGNN_Engine(BaseEngine):
    def __init__(self, **args):
        super(DeepStateGNN_Engine, self).__init__(**args)

    def forward(self, X, label, isTrain = False, query_node = None):
        pred = self.model(X, label)
        pred = torch.stack(pred, dim = 0)
        
        if query_node is not None:
            pred = pred[:, :, query_node, :]
            label = label[:, :, query_node, :]



        return pred, label, None

    