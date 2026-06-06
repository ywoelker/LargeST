"""
SparseStateGNN — observation-level scaling variant of DeepStateGNN.

Instead of processing the full (B, N, T, D) tensor, this model:
1. Extracts only the non-zero measurements → (B, M, ...) where M = max observations in the batch
2. Embeds each measurement individually with sinusoidal positional encoding of its timestep
3. Aggregates measurements into DSN states via static_prefilter assignment weights
4. Runs the same GCN self-attention layers on DSN states
5. Reconstructs predictions for ALL N query sensors via DSN→obs inverse mapping

Forward-pass time should scale with M (available observations) rather than N*T.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.base.model import BaseModel
from src.models.dsgnn import linearized_conv, conv_approximation, linear_kernel


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding (fixed, not learned)
# ---------------------------------------------------------------------------

def sinusoidal_positional_encoding(positions: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Args:
        positions: (*, ) integer or float tensor of timestep indices
        d_model:   embedding dimension (must be even)
    Returns:
        (*, d_model) sinusoidal encoding
    """
    assert d_model % 2 == 0, "d_model must be even for sinusoidal encoding"
    half = d_model // 2
    # frequencies: 1 / 10000^(2i/d_model)
    freq = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=positions.device, dtype=torch.float32) / half
    )  # (half,)
    # outer product
    angles = positions.unsqueeze(-1).float() * freq  # (*, half)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (*, d_model)


class SparseStateGNN(BaseModel):
    """
    Observation-level scaling Deep State GNN.

    Key difference from DeepStateGNN:
    - The input embedding no longer flattens T*3 values per sensor.
    - Instead, each non-zero measurement is processed individually with a
      sinusoidal positional encoding of its timestep index.
    - Measurements are gathered into a sparse list (B, M, ...) where
      M = max non-zero measurements across the batch.
    - Obs→DSN aggregation uses per-measurement static_prefilter weights.
    """

    def __init__(self, num_nodes, in_dim, out_dim, random_feature_dim,
                 time_emb_dim, seq_num, node_emb_dim, use_spatial, dropout,
                 n_contexts, hid_dim, attention_method,
                 time_of_day_size=288, day_of_week_size=7,
                 use_residual=True, use_bn=True, layer_num=3,
                 adding_query_to_dsn=True,
                 pos_enc_dim=16):
        super(SparseStateGNN, self).__init__(num_nodes, in_dim, out_dim)

        self.tau = .25
        self.layer_num = layer_num
        self.in_dim = in_dim
        self.random_feature_dim = random_feature_dim

        self.use_residual = use_residual
        self.use_bn = use_bn

        self.dropout = dropout
        self.activation = nn.ReLU()

        self.time_num = time_of_day_size
        self.week_num = day_of_week_size

        self.num_contexts = n_contexts
        self.node_emb_dim = node_emb_dim
        self.hid_dim = hid_dim

        self.use_spatial = use_spatial
        self.adding_query_to_dsn = adding_query_to_dsn

        self.attention_method = attention_method
        self.pos_enc_dim = pos_enc_dim
        self.seq_num = seq_num

        # ----- learnable DSN embeddings (same as DSGNN) -----
        self.context_emb_layer = nn.Parameter(torch.zeros(self.num_contexts, node_emb_dim))
        nn.init.xavier_uniform_(self.context_emb_layer)

        # ----- time embeddings (same as DSGNN) -----
        self.time_emb_layer = nn.Parameter(torch.zeros(self.time_num, time_emb_dim))
        nn.init.xavier_uniform_(self.time_emb_layer)
        self.week_emb_layer = nn.Parameter(torch.zeros(self.week_num, time_emb_dim))
        nn.init.xavier_uniform_(self.week_emb_layer)

        num_values = 3      # traffic features per timestep
        num_context = in_dim - num_values   # metadata features (35)

        # ----- NEW: per-measurement value embedding -----
        # Each measurement: 1 traffic value + pos_enc_dim sinusoidal encoding
        # We embed traffic values per-feature-channel, then combine with pos enc
        self.value_emb_layer = nn.Sequential(
            nn.Linear(num_values + pos_enc_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
        )

        # ----- context processing (same architecture as DSGNN) -----
        self.context_processing = nn.Sequential(
            nn.Conv2d(num_context, hid_dim, kernel_size=(1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(hid_dim, hid_dim, kernel_size=(1, 1), bias=False),
        )

        # ----- obs → DSN keys/queries (same as DSGNN) -----
        self.W_obs_context_key = nn.Conv2d(hid_dim + time_emb_dim * 2, node_emb_dim, kernel_size=(1, 1), bias=False)
        self.W_obs_context_query = nn.Conv2d(hid_dim + time_emb_dim * 2, node_emb_dim, kernel_size=(1, 1), bias=False)
        self.W_1 = nn.Conv2d(node_emb_dim, node_emb_dim, kernel_size=(1, 1), bias=True)
        self.W_2 = nn.Conv2d(node_emb_dim, node_emb_dim, kernel_size=(1, 1), bias=True)

        # ----- NEW: sparse measurement key projection -----
        # Projects each sparse measurement embedding to the key space for obs→DSN attention
        self.sparse_key_proj = nn.Linear(hid_dim, node_emb_dim)

        # ----- obs → DSN linearized conv -----
        self.linear_obs_2_dsn_conv = linearized_conv(
            hid_dim + 2 * time_emb_dim, hid_dim, self.dropout, self.attention_method,
            self.tau, self.random_feature_dim, non_linearity=True, key_dim=node_emb_dim
        )

        # ----- GCN layers on DSN (identical to DSGNN) -----
        self.linear_conv = nn.ModuleList()
        self.bn = nn.ModuleList()

        for _ in range(self.layer_num):
            self.linear_conv.append(linearized_conv(
                hid_dim + node_emb_dim, hid_dim + node_emb_dim, self.dropout,
                self.attention_method, self.tau, self.random_feature_dim,
                non_linearity=False, key_dim=node_emb_dim
            ))
            self.bn.append(nn.LayerNorm(hid_dim + node_emb_dim))

        # ----- DSN → obs (identical to DSGNN) -----
        self.hid_dim_times_after_conv = 1
        self.linear_dsn_2_obs_conv = linearized_conv(
            (node_emb_dim + hid_dim) * 2, hid_dim * self.hid_dim_times_after_conv,
            self.dropout, self.attention_method, self.tau, self.random_feature_dim,
            key_dim=node_emb_dim
        )

        self.bn_obs_to_context = nn.LayerNorm(hid_dim)
        self.bn_context_to_obs = nn.LayerNorm(hid_dim * self.hid_dim_times_after_conv)

        # ----- Regression (same as DSGNN) -----
        # Skip connection uses context + time emb (no traffic values for query)
        self.regression_layer = nn.Conv2d(
            hid_dim * (self.hid_dim_times_after_conv + 1) + 2 * time_emb_dim,
            out_dim, kernel_size=(1, 1), bias=True
        )

    # ------------------------------------------------------------------
    # Sparse extraction: (B, N, T, D) → (B, M, ...) padded to max
    # ------------------------------------------------------------------

    def _extract_sparse(self, x, input_mask):
        """
        Extract non-zero measurements into a padded sparse representation.

        Args:
            x: (B, N, T, D) full input tensor
            input_mask: (B, T, N, 1) binary mask — 1 where measurement exists.
                        If None, treat all measurements as present.

        Returns:
            meas_values: (B, M, 3) traffic values of non-zero measurements
            meas_sensor_idx: (B, M) sensor index of each measurement (long)
            meas_time_idx: (B, M) timestep index of each measurement (long)
            meas_mask: (B, M) boolean mask — True for valid measurements
            M: int, max measurements in this batch
        """
        B, N, T, D = x.size()

        if input_mask is not None:
            # input_mask is (B, T, N, 1) → reshape to (B, N, T)
            mask_bnt = input_mask.squeeze(-1).permute(0, 2, 1)  # (B, N, T)
        else:
            # Infer mask from traffic values: non-zero if any of the 3 traffic features is non-zero
            mask_bnt = (x[..., :3].abs().sum(dim=-1) > 0)  # (B, N, T)

        traffic_vals = x[..., :3]  # (B, N, T, 3)

        # Flatten N*T per batch element and work in 2-D to avoid per-batch loops
        mask_flat = mask_bnt.reshape(B, N * T)            # (B, N*T)
        counts = mask_flat.sum(dim=1)                      # (B,)
        M = int(counts.max().item())
        M = max(M, 1)  # at least 1 to avoid empty tensors

        # Argsort so that True entries come first per row (descending)
        # On MPS, sort is supported; argsort on bool isn't — cast to int first
        order = mask_flat.int().neg().argsort(dim=1, stable=True)  # (B, N*T)
        order_M = order[:, :M]  # (B, M) — top-M indices per batch

        # Recover (sensor, time) from flat index
        meas_sensor_idx = order_M // T   # (B, M)
        meas_time_idx = order_M % T      # (B, M)

        # Gather traffic values
        # Expand indices for gather: need (B, M, 3)
        b_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, M)
        meas_values = traffic_vals[b_idx, meas_sensor_idx, meas_time_idx, :]  # (B, M, 3)

        # Validity mask
        meas_mask = torch.arange(M, device=x.device).unsqueeze(0) < counts.unsqueeze(1)  # (B, M)

        # Zero out padding positions
        meas_values = meas_values * meas_mask.unsqueeze(-1).float()

        return meas_values, meas_sensor_idx, meas_time_idx, meas_mask, M

    # ------------------------------------------------------------------
    # Sparse obs → DSN aggregation
    # ------------------------------------------------------------------

    def _sparse_obs_to_dsn(self, meas_emb, meas_sensor_idx, meas_mask,
                           static_prefilter, dsn_queries):
        """
        Aggregate sparse measurement embeddings into DSN states using
        static_prefilter assignment weights and learned attention.

        Args:
            meas_emb: (B, M, hid_dim) embedded measurements
            meas_sensor_idx: (B, M) sensor index per measurement
            meas_mask: (B, M) boolean mask
            static_prefilter: (N, C) assignment matrix
            dsn_queries: (B, C, node_emb_dim) DSN query embeddings

        Returns:
            dsn_states: (B, C, hid_dim) aggregated DSN states
            attention_scores: None (placeholder for compatibility)
        """
        B, M, D = meas_emb.shape
        C = self.num_contexts

        # Project measurement embeddings to key space
        meas_keys = self.sparse_key_proj(meas_emb)  # (B, M, node_emb_dim)

        # Compute attention: query-key dot product
        # dsn_queries: (B, C, node_emb_dim), meas_keys: (B, M, node_emb_dim)
        attn_logits = torch.bmm(dsn_queries, meas_keys.transpose(1, 2))  # (B, C, M)
        attn_logits = attn_logits / math.sqrt(self.node_emb_dim)

        # Apply static_prefilter as a hard mask: each measurement from sensor n
        # can only attend to DSN c if static_prefilter[n, c] > 0
        # static_prefilter: (N, C), meas_sensor_idx: (B, M)
        # Gather assignment weights: (B, M, C)
        sp_weights = static_prefilter[meas_sensor_idx]  # (B, M, C)
        sp_mask = (sp_weights > 0).permute(0, 2, 1)  # (B, C, M)

        # Combine: mask invalid assignments and padding
        # meas_mask: (B, M) → (B, 1, M)
        combined_mask = sp_mask & meas_mask.unsqueeze(1)  # (B, C, M)

        attn_logits = attn_logits.masked_fill(~combined_mask, float('-inf'))

        # Softmax over measurements dimension
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, C, M)
        # Replace NaN from all-inf rows with 0
        attn_weights = attn_weights.masked_fill(~combined_mask, 0.0)

        # Weight by static_prefilter values too
        sp_weights_transposed = sp_weights.permute(0, 2, 1)  # (B, C, M)
        attn_weights = attn_weights * sp_weights_transposed

        # Re-normalize
        attn_sum = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        attn_weights = attn_weights / attn_sum

        # Aggregate: (B, C, M) × (B, M, hid_dim) → (B, C, hid_dim)
        dsn_states = torch.bmm(attn_weights, meas_emb)

        return dsn_states, None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x, feat=None, static_prefilter=None,
                valid_observations=None, query_index=None, input_mask=None):
        """
        Args:
            x: (B, N, T, D) input tensor — same as DSGNN
            static_prefilter: (N, C) or (N_available, C) assignment matrix
            input_mask: (B, T, N, 1) binary mask; 1 = measurement present.
                        If None, infer from non-zero traffic values.
            Other args: same as DSGNN for compatibility.
        Returns:
            dict with 'prediction', 'assignment_scores_source/target', 'dsn_states'
        """
        B, N, T, D = x.size()

        # ----- 1. Extract time/week embeddings from last timestep (same as DSGNN) -----
        time_emb = self.time_emb_layer[(x[:, :, -1, 1] * self.time_num).long()]
        week_emb = self.week_emb_layer[x[:, :, -1, 2].long()]

        # ----- 2. Separate context and values -----
        x_context = x[..., -1, 3:]   # (B, N, num_context=35)
        # x_value = x[..., :3]         # (B, N, T, 3) — used via sparse extraction

        # ----- 3. Extract sparse measurements -----
        meas_values, meas_sensor_idx, meas_time_idx, meas_mask, M = \
            self._extract_sparse(x, input_mask)
        # meas_values: (B, M, 3), meas_sensor/time_idx: (B, M), meas_mask: (B, M)

        # ----- 4. Embed each measurement with sinusoidal pos encoding -----
        pos_enc = sinusoidal_positional_encoding(meas_time_idx, self.pos_enc_dim)  # (B, M, pos_enc_dim)
        meas_input = torch.cat([meas_values, pos_enc], dim=-1)  # (B, M, 3 + pos_enc_dim)
        meas_emb = self.value_emb_layer(meas_input)  # (B, M, hid_dim)
        # Zero out padded measurements
        meas_emb = meas_emb * meas_mask.unsqueeze(-1).float()

        # ----- 5. Context processing for ALL sensors (query side) -----
        x_context_4d = x_context.transpose(1, 2).unsqueeze(-1)  # (B, 35, N, 1)
        x_context_emb = self.context_processing(x_context_4d)   # (B, hid_dim, N, 1)

        time_emb_4d = time_emb.transpose(1, 2).unsqueeze(-1)    # (B, time_emb_dim, N, 1)
        week_emb_4d = week_emb.transpose(1, 2).unsqueeze(-1)    # (B, time_emb_dim, N, 1)

        # x_g: query context for all N sensors (context + time + week)
        x_g = torch.cat([x_context_emb, time_emb_4d, week_emb_4d], dim=1)  # (B, hid+2*te, N, 1)

        # ----- 6. Sparse obs → DSN aggregation -----
        # DSN queries
        queries = self.context_emb_layer.unsqueeze(0).expand(B, -1, -1)  # (B, C, node_emb_dim)
        raw_dsn = queries.clone()  # (B, C, node_emb_dim) — for logging

        # Aggregate sparse measurements into DSN states
        if static_prefilter is not None:
            dsn_states, assignment_scores_source = self._sparse_obs_to_dsn(
                meas_emb, meas_sensor_idx, meas_mask,
                static_prefilter, queries
            )
        else:
            # Without static_prefilter, use uniform assignment
            # (B, M, hid_dim) → mean-pool to (B, 1, hid_dim) → expand to (B, C, hid_dim)
            masked_emb = meas_emb * meas_mask.unsqueeze(-1).float()
            n_valid = meas_mask.sum(dim=-1, keepdim=True).clamp(min=1).unsqueeze(-1)  # (B, 1, 1)
            dsn_states = masked_emb.sum(dim=1, keepdim=True) / n_valid  # (B, 1, hid_dim)
            dsn_states = dsn_states.expand(-1, self.num_contexts, -1)
            assignment_scores_source = None

        # Apply layer norm
        dsn_states = self.bn_obs_to_context(dsn_states)  # (B, C, hid_dim)

        obs_augmented_dsn = dsn_states.clone()  # (B, C, hid_dim) — for logging

        # ----- 7. GCN layers on DSN (identical to DSGNN) -----
        # Reshape to (B, hid_dim, C, 1) for Conv2d-based layers
        deepstate = dsn_states.permute(0, 2, 1).unsqueeze(-1)  # (B, hid_dim, C, 1)

        # Cat with DSN embeddings
        queries_4d = self.context_emb_layer.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1)
        # queries_4d: (B, node_emb_dim, C, 1)

        node_vec1 = self.W_1(queries_4d)  # (B, node_emb_dim, C, 1)
        node_vec2 = self.W_2(queries_4d)  # (B, node_emb_dim, C, 1)
        node_vec1 = node_vec1.permute(0, 2, 3, 1)  # (B, C, 1, node_emb_dim)
        node_vec2 = node_vec2.permute(0, 2, 3, 1)  # (B, C, 1, node_emb_dim)

        if self.adding_query_to_dsn:
            deepstate = torch.cat([deepstate, queries_4d], dim=1)  # (B, hid+emb, C, 1)
        else:
            deepstate = torch.cat([deepstate, torch.zeros_like(queries_4d)], dim=1)

        deepstate_pool = [deepstate]

        for i in range(self.layer_num):
            if self.use_residual:
                residual = deepstate
            deepstate, _, _, _ = self.linear_conv[i](deepstate, node_vec1, node_vec2, None)

            if self.use_residual:
                deepstate = deepstate + residual

            if self.use_bn:
                deepstate = deepstate.permute(0, 2, 3, 1)
                deepstate = self.bn[i](deepstate)
                deepstate = deepstate.permute(0, 3, 1, 2)

        deepstate_pool.append(deepstate)
        deepstate = torch.cat(deepstate_pool, dim=1)  # (B, 2*(hid+emb), C, 1)

        gnn_convolved_dsn = deepstate.permute(0, 2, 1, 3).squeeze(-1)  # (B, C, 2*(hid+emb))

        # ----- 8. DSN → obs reconstruction (identical to DSGNN) -----
        if query_index is not None:
            x_g_query = x_g[:, :, query_index:query_index + 1, :]
            sp_inv = static_prefilter[query_index:query_index + 1, :] if static_prefilter is not None else None
        else:
            x_g_query = x_g
            sp_inv = static_prefilter

        queries_obs = self.W_obs_context_query(x_g_query)
        queries_obs = queries_obs.permute(0, 2, 3, 1)  # (B, N_q, 1, node_emb_dim)

        keys_dsn = self.context_emb_layer.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1)
        keys_dsn = keys_dsn.permute(0, 2, 3, 1)  # (B, C, 1, node_emb_dim)

        if sp_inv is not None:
            x_out, _, _, assignment_scores_target = self.linear_dsn_2_obs_conv(
                deepstate, queries_obs, keys_dsn, sp_inv.T
            )
        else:
            x_out, _, _, assignment_scores_target = self.linear_dsn_2_obs_conv(
                deepstate, queries_obs, keys_dsn, None
            )

        x_out = x_out.permute(0, 2, 3, 1)  # (B, N_q, 1, hid)
        x_out = self.bn_context_to_obs(x_out)
        x_out = x_out.permute(0, 3, 1, 2)  # (B, hid, N_q, 1)

        # ----- 9. Skip connection + regression -----
        # Skip: use query context (context_emb + time + week) — no traffic values for query
        x_skip = x_g_query  # (B, hid+2*te, N_q, 1)

        x_cat = torch.cat([x_skip, x_out], dim=1)  # (B, hid*2 + 2*te, N_q, 1)
        x_cat = self.activation(x_cat)

        x_pred = self.regression_layer(x_cat)  # (B, out_dim, N_q, 1)
        x_pred = x_pred.squeeze(-1).permute(0, 2, 1)  # (B, N_q, out_dim)

        return {
            "prediction": x_pred.transpose(1, 2).unsqueeze(-1),
            "assignment_scores_source": assignment_scores_source,
            "assignment_scores_target": assignment_scores_target,
            "dsn_states": {
                "raw": raw_dsn,
                "obs_augmented": obs_augmented_dsn,
                "gnn_convolved": gnn_convolved_dsn,
            },
        }
