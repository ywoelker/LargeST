import torch
import numpy as np
from tqdm import tqdm
from src.base.engine import BaseEngine


class DSGNN_Engine(BaseEngine):
    def __init__(self, static_prefilter, additional_loss_weight, **args):
        super(DSGNN_Engine, self).__init__(**args)

        self.static_prefilter = static_prefilter
        self.additional_loss_weight = additional_loss_weight

        self.embedding_evolution = {}

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

        if pred_dict['assignment_scores_source'] is None or pred_dict['assignment_scores_target'] is None:
            return loss

        mean_source_attention = torch.norm(pred_dict['assignment_scores_source'], p = 1)
        mean_target_attention = torch.norm(pred_dict['assignment_scores_target'], p = 1)

        additional_loss = self.additional_loss_weight * ( mean_source_attention + mean_target_attention)

        if self.static_prefilter is not None:
            additional_loss /= self.model.num_contexts


        return loss + additional_loss
    

    def evaluate(self, mode) -> tuple:
        evaluation_results = super().evaluate(mode)

        if self.epoch in self.embedding_evolution or mode != 'val':
            return evaluation_results
        
        self.embedding_evolution[self.epoch] = self.model.context_emb_layer.detach().cpu().numpy()


        return evaluation_results
    
    def train(self):
        train_result = super().train()

        for keys in list(self.embedding_evolution.keys()):
            if keys > self.best_epoch:
                del self.embedding_evolution[keys]

        embedding_evolution_array = []
        for epoch in range(self.best_epoch + 1):
            embedding_evolution_array.append(self.embedding_evolution[epoch])

        np.save(self._save_path + '/embedding_evolution.npy', np.array(embedding_evolution_array))

        return train_result        
