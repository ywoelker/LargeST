import torch
import numpy as np
from tqdm import tqdm
from src.base.engine import BaseEngine


class DSGNN_Engine(BaseEngine):
    def __init__(self, static_prefilter, additional_loss_weight, **args):
        super(DSGNN_Engine, self).__init__(**args)

        self.static_prefilter = static_prefilter
        self.additional_loss_weight = additional_loss_weight

    def forward(self, X, label, isTrain = False):

        if self.static_prefilter is not None:
            static_assignment = self.static_prefilter
            
            if self.current_available_sensors is not None:
                static_assignment = static_assignment[self.current_available_sensors.squeeze() == 1]
            
        else:
            static_assignment = None


        pred_dict = self.model(X.transpose(1,2), label, static_prefilter = static_assignment)
        pred = pred_dict['prediction']


        return pred, label, pred_dict
    

    def loss(self, pred, label, mask_value, loss_container):
        pred_dict = loss_container

        loss = super(DSGNN_Engine, self).loss(pred, label, mask_value, loss_container)

        mean_source_attention = torch.norm(pred_dict['assignment_scores_source'], p = 1)
        mean_target_attention = torch.norm(pred_dict['assignment_scores_target'], p = 1)

        additional_loss = self.additional_loss_weight * ( mean_source_attention + mean_target_attention)

        if self.static_prefilter is not None:
            additional_loss /= self.model.num_contexts


        return loss + additional_loss

        
