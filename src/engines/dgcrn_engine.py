import torch
import numpy as np
from src.base.engine import BaseEngine
from src.utils.metrics import masked_mape, masked_rmse

class DGCRN_Engine(BaseEngine):
    def __init__(self, step_size, horizon, **args):
        super(DGCRN_Engine, self).__init__(**args)
        self._step_size = step_size
        self._horizon = horizon
        self._task_level = 0


    def forward(self, X, label, isTrain = False):

        if self._iter_cnt % self._step_size == 0 and self._task_level < self._horizon:
                self._task_level += 1

        if isTrain:
            pred = self.model(X, label, self._iter_cnt, self._task_level)
            pred = pred[:, :self._task_level, :, :]
            label = label[:, :self._task_level, :, :]

        else: 
            pred = self.model(X, label)

        return pred, label, None
