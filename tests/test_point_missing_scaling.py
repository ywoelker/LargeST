"""
Benchmark: DSGNN vs SparseStateGNN forward-pass time under point-missing scenarios.

Demonstrates that:
- DSGNN does NOT scale with available observations (constant forward-pass time)
- SparseStateGNN DOES scale with available observations (time decreases with more missing data)
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.dsgnn import DeepStateGNN
from src.models.sparsestategnn import SparseStateGNN
from src.utils.dataloader import StandardScaler

SEED = 42
DEVICE = "cpu"
SEQ_LEN = 12
HORIZON = 12
BATCH_SIZE = 8
NUM_SENSORS_SUBSETS = [100, 500]
NUM_BATCHES = 10
MISSING_RATES = [0.25, 0.50, 0.75]
WARMUP_BATCHES = 2


def load_sd_subset(num_sensors):
    data_path = os.path.join("data", "sd", "2019", "his.npz")
    ptr = np.load(data_path, allow_pickle=True)
    data = ptr["data"]  # (T, N, 3)
    metadata = ptr["metadata"]  # (N, 35)
    mean, std = float(ptr["mean"]), float(ptr["std"])

    sa = np.load(os.path.join("data", "sd", "2019", "static_assignment.npz"))
    static_assignment = sa["static_assignment_matrices"]  # (N, C)

    rng = np.random.RandomState(SEED)
    sensor_idx = rng.choice(data.shape[1], size=num_sensors, replace=False)
    sensor_idx.sort()

    data = data[:, sensor_idx, :]
    metadata = metadata[sensor_idx, :]
    static_assignment = static_assignment[sensor_idx, :]

    data[..., :1] = (data[..., :1] - mean) / std
    scaler = StandardScaler(mean, std)
    return data, metadata, static_assignment, scaler


INPUT_DIM = 38  # 3 traffic features + 35 metadata features


def build_dsgnn(num_sensors, n_contexts):
    model = DeepStateGNN(
        num_nodes=num_sensors,
        seq_num=SEQ_LEN,
        in_dim=INPUT_DIM,
        out_dim=HORIZON,
        random_feature_dim=64,
        node_emb_dim=32,
        layer_num=3,
        adding_query_to_dsn=True,
        time_emb_dim=8,
        use_residual=True,
        use_bn=True,
        use_spatial=False,
        hid_dim=16,
        n_contexts=n_contexts,
        dropout=0.0,
        attention_method="MLA",
        time_of_day_size=288,
        day_of_week_size=7,
    )
    model.to(DEVICE).eval()
    return model


def build_sparse_stategnn(num_sensors, n_contexts):
    model = SparseStateGNN(
        num_nodes=num_sensors,
        seq_num=SEQ_LEN,
        in_dim=INPUT_DIM,
        out_dim=HORIZON,
        random_feature_dim=64,
        node_emb_dim=32,
        layer_num=3,
        adding_query_to_dsn=True,
        time_emb_dim=8,
        use_residual=True,
        use_bn=True,
        use_spatial=False,
        hid_dim=16,
        n_contexts=n_contexts,
        dropout=0.0,
        attention_method="MLA",
        time_of_day_size=288,
        day_of_week_size=7,
        pos_enc_dim=16,
    )
    model.to(DEVICE).eval()
    return model


def make_batch(data, metadata, start_idx, missing_rate, rng):
    """Build a batch with point-missing applied to traffic values.

    Concatenates metadata to match the dataloader's behaviour (input_dim=38).
    Returns:
        source: (B, T, N, 38) tensor
        target: (B, T_horizon, N, 1) tensor
        input_mask: (B, T, N, 1) binary mask (1 = present)
    """
    n_sensors = data.shape[1]

    sources, targets, masks = [], [], []
    for b in range(BATCH_SIZE):
        t0 = start_idx + b
        x_slice = data[t0 + np.arange(SEQ_LEN), :, :].copy()  # (T, N, 3)
        y_slice = data[t0 + SEQ_LEN + np.arange(HORIZON), :, :1]

        if missing_rate > 0:
            mask = (rng.random((SEQ_LEN, n_sensors)) > missing_rate).astype(np.float32)
            x_slice[..., :1] *= mask[..., np.newaxis]
        else:
            mask = np.ones((SEQ_LEN, n_sensors), dtype=np.float32)

        # tile metadata across timesteps and concat (same as dataloader)
        meta_tile = np.tile(metadata, (SEQ_LEN, 1, 1))  # (T, N, 35)
        x_full = np.concatenate([x_slice, meta_tile], axis=-1)  # (T, N, 38)

        sources.append(x_full)
        targets.append(y_slice)
        masks.append(mask[..., np.newaxis])  # (T, N, 1)

    source = torch.tensor(np.stack(sources), dtype=torch.float32).to(DEVICE)
    target = torch.tensor(np.stack(targets), dtype=torch.float32).to(DEVICE)
    input_mask = torch.tensor(np.stack(masks), dtype=torch.float32).to(DEVICE)
    return source, target, input_mask


def benchmark_model(model, model_name, data, metadata, static_assignment, num_sensors, is_sparse=False):
    rng = np.random.RandomState(SEED)
    valid_start = SEQ_LEN
    valid_end = data.shape[0] - HORIZON - BATCH_SIZE - 1

    sa_tensor = torch.tensor(static_assignment, dtype=torch.float32).to(DEVICE)

    results = {}
    for rate in MISSING_RATES:
        offsets = rng.randint(valid_start, valid_end, size=NUM_BATCHES + WARMUP_BATCHES)
        times = []

        for i, off in enumerate(offsets):
            source, target, input_mask = make_batch(data, metadata, off, rate, rng)
            # Both models expect (B, N, T, D)
            x = source.transpose(1, 2)  # → (B, N, T, F)

            with torch.no_grad():
                t0 = time.perf_counter()
                if is_sparse:
                    _ = model(x, static_prefilter=sa_tensor, input_mask=input_mask)
                else:
                    _ = model(x, static_prefilter=sa_tensor)
                t1 = time.perf_counter()

            if i >= WARMUP_BATCHES:
                times.append(t1 - t0)

        mean_t = np.mean(times)
        std_t = np.std(times)
        obs_per_batch = int(BATCH_SIZE * num_sensors * SEQ_LEN * (1 - rate))
        results[rate] = (mean_t, std_t, obs_per_batch)

        print(
            f"  {model_name:>20s} | N={num_sensors:>4d} | Missing {rate*100:5.1f}% | "
            f"~{obs_per_batch:>6d} non-zero obs/batch | "
            f"fwd {mean_t:.4f} +/- {std_t:.4f} s"
        )

    return results


def print_summary(all_results, model_name):
    print(f"\n  {model_name}:")
    for num_sensors in NUM_SENSORS_SUBSETS:
        if num_sensors not in all_results:
            continue
        results = all_results[num_sensors]
        rates = sorted(results.keys())
        t_25 = results[rates[0]][0]
        t_75 = results[rates[-1]][0]
        ratio = t_25 / t_75 if t_75 > 0 else float("inf")

        print(f"\n    N = {num_sensors}:")
        for rate in MISSING_RATES:
            t, s, obs = results[rate]
            print(f"      Missing {rate*100:4.0f}%: {t:.4f} +/- {s:.4f} s  (~{obs} obs)")

        print(
            f"      Scaling ratio (time@25% / time@75%): {ratio:.2f}x"
        )
    return ratio


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)

    dsgnn_results = {}
    sparse_results = {}

    for num_sensors in NUM_SENSORS_SUBSETS:
        print(f"\n{'#' * 80}")
        print(f"# Benchmark — N = {num_sensors} sensors")
        print(f"{'#' * 80}")

        data, metadata, static_assignment, scaler = load_sd_subset(num_sensors)
        n_contexts = static_assignment.shape[1]
        print(f"  data shape: {data.shape}  |  metadata: {metadata.shape}  |  static_assignment: {static_assignment.shape}")

        # --- DSGNN ---
        print(f"\n  --- DSGNN (baseline) ---")
        dsgnn_model = build_dsgnn(num_sensors, n_contexts)
        n_params = sum(p.numel() for p in dsgnn_model.parameters())
        print(f"  #params: {n_params:,}")
        print(f"  batch_size={BATCH_SIZE}, batches={NUM_BATCHES}, warmup={WARMUP_BATCHES}")
        print("-" * 80)

        dsgnn_results[num_sensors] = benchmark_model(
            dsgnn_model, "DSGNN", data, metadata, static_assignment, num_sensors, is_sparse=False
        )
        del dsgnn_model

        # --- SparseStateGNN ---
        print(f"\n  --- SparseStateGNN (observation-level scaling) ---")
        sparse_model = build_sparse_stategnn(num_sensors, n_contexts)
        n_params = sum(p.numel() for p in sparse_model.parameters())
        print(f"  #params: {n_params:,}")
        print(f"  batch_size={BATCH_SIZE}, batches={NUM_BATCHES}, warmup={WARMUP_BATCHES}")
        print("-" * 80)

        sparse_results[num_sensors] = benchmark_model(
            sparse_model, "SparseStateGNN", data, metadata, static_assignment, num_sensors, is_sparse=True
        )
        del sparse_model

    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")

    dsgnn_ratio = print_summary(dsgnn_results, "DSGNN")
    sparse_ratio = print_summary(sparse_results, "SparseStateGNN")

    print(
        "\n  DSGNN: A ratio close to 1.0 confirms the model does NOT scale "
        "with the number of available observations."
    )
    print(
        "  SparseStateGNN: A ratio > 1.0 means the model IS faster "
        "when more data is missing (scales with available observations)."
    )


if __name__ == "__main__":
    main()
