import torch
import numpy as np
from src.base.engine import BaseEngine

class D2STGNN_Engine(BaseEngine):
    def __init__(self, cl_step, warm_step, horizon, **args):
        super(D2STGNN_Engine, self).__init__(**args)
        self._cl_step = cl_step
        self._warm_step = warm_step
        self._horizon = horizon
        self._cl_len = 0


    def forward(self, X, label, isTrain = False):
        pred = self.model(X, label)

        if self._iter_cnt < self._warm_step:
            self._cl_len = self._horizon
        elif self._iter_cnt == self._warm_step:
            self._cl_len = 1
        else:
            if (self._iter_cnt - self._warm_step) % self._cl_step == 0 and self._cl_len < self._horizon:
                self._cl_len += 1

        if isTrain:
            pred = pred[:, :self._cl_len, :, :]
            label = label[:, :self._cl_len, :, :]   


        return pred, label, None
