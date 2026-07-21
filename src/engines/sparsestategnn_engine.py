"""
Engine for SparseStateGNN — extends DSGNN_Engine to pass input_mask to the model.
"""

import torch
import numpy as np
from src.engines.dsgnn_engine import DSGNN_Engine


class SparseStateGNN_Engine(DSGNN_Engine):
    """
    Thin wrapper around DSGNN_Engine that forwards the current input mask
    (current_x_mask) to SparseStateGNN so the model knows which measurements
    are present vs. missing.
    """

    def forward(self, X, label, isTrain=False, **kwargs):
        # Prepare static_prefilter (same logic as DSGNN_Engine)
        if self.static_prefilter is not None:
            static_assignment = self.static_prefilter
            if self.current_available_sensors is not None:
                static_assignment = static_assignment[self.current_available_sensors.squeeze() == 1]
        else:
            static_assignment = None

        # Get the input mask — may be None if not set
        input_mask = getattr(self, 'current_x_mask', None)

        # SparseStateGNN expects (B, N, T, D) — X comes in as (B, T, N, F)
        x_transposed = X.transpose(1, 2)  # → (B, N, T, F)

        if kwargs.get('query_node', None) is not None:
            query_node = kwargs['query_node']
            pred_dict = self.model(
                x_transposed, label,
                static_prefilter=static_assignment,
                query_index=query_node,
                input_mask=input_mask
            )
        else:
            pred_dict = self.model(
                x_transposed, label,
                static_prefilter=static_assignment,
                input_mask=input_mask
            )

        pred = pred_dict['prediction']
        return pred, label, pred_dict
