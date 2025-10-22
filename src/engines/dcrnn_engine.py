import torch
import numpy as np
from tqdm import tqdm
from src.base.engine import BaseEngine

class DCRNN_Engine(BaseEngine):
    def __init__(self, **args):
        super(DCRNN_Engine, self).__init__(**args)


    def forward(self, X, label, isTrain = False ):
        pred = self.model(X, label, self._iter_cnt)

        return pred, label, None