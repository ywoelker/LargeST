import torch
import numpy as np
from tqdm import tqdm
from src.base.engine import BaseEngine


class DSGNN_Engine(BaseEngine):
    def __init__(self, **args):
        super(DSGNN_Engine, self).__init__(**args)


    def forward(self, X, label, isTrain = False):
        pred_dict = self.model(X.transpose(1,2), label)
        pred = pred_dict['prediction']


        return pred, label, pred_dict
    

    def loss(self, pred, label, mask_value, loss_container):
        pred_dict = loss_container

        loss = super(DSGNN_Engine, self).loss(pred, label, mask_value, loss_container)

        mean_source_attention = torch.norm(pred_dict['assignment_scores_source'], p = 1)
        mean_target_attention = torch.norm(pred_dict['assignment_scores_target'], p = 1)

        return loss +   .001 * mean_source_attention + .001 * mean_target_attention
        
