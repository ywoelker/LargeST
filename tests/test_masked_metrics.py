"""
Tests for masked metrics (MAE, MSE, RMSE, MAPE) to verify that:
1. NaN labels are excluded from the denominator (not just zeroed in the numerator)
2. null_val-masked positions are excluded from the denominator (with atol=0.5)
3. label_mask further restricts which positions contribute
4. The result matches a naive hand-computed mean over valid positions only
"""

import os
import sys
import torch
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.metrics import (
    _label_mask,
    masked_mae,
    masked_mse,
    masked_rmse,
    masked_mape,
    compute_all_metrics,
)

ATOL = 1e-5
MASK_ATOL = 1e-5  # must match _label_mask default


def valid_mask(labels, null_val):
    """Reproduce _label_mask logic: exclude NaN and labels within atol of null_val."""
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = torch.abs(labels - null_val) > MASK_ATOL
    if torch.isnan(labels).any():
        mask = mask & ~torch.isnan(labels)
    return mask


# ── helpers ──────────────────────────────────────────────────────────────

def naive_mae(preds, labels, valid):
    return torch.abs(preds[valid] - labels[valid]).mean()


def naive_mse(preds, labels, valid):
    return ((preds[valid] - labels[valid]) ** 2).mean()


def naive_mape(preds, labels, valid):
    return (torch.abs(preds[valid] - labels[valid]) / labels[valid]).mean()


# ── tests ────────────────────────────────────────────────────────────────

def test_no_nan_no_nullval():
    """All positions valid — metrics should equal plain torch mean."""
    preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
    labels = torch.tensor([1.5, 2.5, 3.5, 4.5])
    null_val = torch.tensor(float("nan"))

    expected_mae = torch.abs(preds - labels).mean()
    assert torch.isclose(masked_mae(preds, labels, null_val), expected_mae, atol=ATOL), \
        "MAE should match plain mean when all positions are valid"
    print("  PASS test_no_nan_no_nullval")


def test_null_val_excluded():
    """Positions within atol of null_val should be excluded from mean."""
    preds = torch.tensor([10.0, 20.0, 30.0, 0.0])
    labels = torch.tensor([12.0, 22.0, 32.0, 0.0])
    null_val = torch.tensor(0.0)

    v = valid_mask(labels, null_val)
    expected = naive_mae(preds, labels, v)
    result = masked_mae(preds, labels, null_val)
    assert torch.isclose(result, expected, atol=ATOL), \
        f"MAE should average only over non-null positions: got {result:.6f}, expected {expected:.6f}"
    print("  PASS test_null_val_excluded")


def test_null_val_tolerance():
    """Labels within atol=1e-5 of null_val should also be excluded."""
    preds = torch.tensor([10.0, 20.0, 30.0, 40.0])
    labels = torch.tensor([12.0, 22.0, 0.000003, 42.0])  # 0.000003 is within 1e-5 of null_val=0
    null_val = torch.tensor(0.0)

    v = valid_mask(labels, null_val)
    assert v.sum() == 3, "Label 0.000003 should be excluded (within atol=1e-5 of 0)"
    expected = naive_mae(preds, labels, v)
    result = masked_mae(preds, labels, null_val)
    assert torch.isclose(result, expected, atol=ATOL), \
        f"Tolerance test: got {result:.6f}, expected {expected:.6f}"
    print("  PASS test_null_val_tolerance")


def test_nan_labels_excluded_from_denominator():
    """
    Core test: 75% of labels are NaN (simulating point_missing_075).
    The reported MAE must equal the MAE computed only over valid positions.
    """
    torch.manual_seed(42)
    N = 1000
    preds = torch.randn(N) * 20 + 50
    labels = torch.randn(N) * 20 + 50

    mask = torch.rand(N) > 0.75
    labels_nan = labels.clone()
    labels_nan[~mask] = float("nan")

    null_val = labels_nan[~labels_nan.isnan()].min()
    v = valid_mask(labels_nan, null_val)

    expected = naive_mae(preds, labels_nan, v)
    result = masked_mae(preds, labels_nan, null_val)

    assert torch.isclose(result, expected, atol=ATOL), (
        f"With 75% NaN labels, MAE denominator must count only valid positions.\n"
        f"  Got:      {result:.4f}\n"
        f"  Expected: {expected:.4f}\n"
        f"  Ratio:    {result / expected:.4f}x"
    )
    print("  PASS test_nan_labels_excluded_from_denominator")


def test_nan_labels_excluded_mse():
    """Same as above but for MSE."""
    torch.manual_seed(42)
    N = 400
    preds = torch.randn(N) * 10
    labels = torch.randn(N) * 10

    mask = torch.rand(N) > 0.5
    labels_nan = labels.clone()
    labels_nan[~mask] = float("nan")

    null_val = labels_nan[~labels_nan.isnan()].min()
    v = valid_mask(labels_nan, null_val)

    expected = naive_mse(preds, labels_nan, v)
    result = masked_mse(preds, labels_nan, null_val)

    assert torch.isclose(result, expected, atol=1e-3), \
        f"MSE with NaN labels: got {result:.4f}, expected {expected:.4f}"
    print("  PASS test_nan_labels_excluded_mse")


def test_nan_labels_excluded_mape():
    """Same for MAPE, using strictly positive labels far from null_val."""
    torch.manual_seed(42)
    N = 400
    labels = torch.rand(N) * 100 + 10  # all positive, well above atol
    preds = labels + torch.randn(N) * 5

    mask = torch.rand(N) > 0.6
    labels_nan = labels.clone()
    labels_nan[~mask] = float("nan")

    null_val = labels_nan[~labels_nan.isnan()].min()
    v = valid_mask(labels_nan, null_val)

    expected = naive_mape(preds, labels_nan, v)
    result = masked_mape(preds, labels_nan, null_val)

    assert torch.isclose(result, expected, atol=1e-3), \
        f"MAPE with NaN labels: got {result:.4f}, expected {expected:.4f}"
    print("  PASS test_nan_labels_excluded_mape")


def test_label_mask_further_restricts():
    """label_mask=0 positions should also be excluded from the denominator."""
    preds = torch.tensor([10.0, 20.0, 30.0, 40.0])
    labels = torch.tensor([12.0, 22.0, 32.0, 42.0])
    null_val = torch.tensor(float("nan"))
    label_mask = torch.tensor([1.0, 1.0, 0.0, 0.0])

    expected = torch.abs(preds[:2] - labels[:2]).mean()
    result = masked_mae(preds, labels, null_val, label_mask=label_mask)

    assert torch.isclose(result, expected, atol=ATOL), \
        f"label_mask should restrict to first 2 positions: got {result:.4f}, expected {expected:.4f}"
    print("  PASS test_label_mask_further_restricts")


def test_nan_plus_label_mask():
    """Combined: NaN labels AND label_mask, both exclude positions."""
    preds = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    labels = torch.tensor([12.0, float("nan"), 32.0, float("nan"), 52.0])
    null_val = torch.tensor(0.0)
    label_mask = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])

    # Position 0: valid (12, not NaN, mask=1, |12-0|>0.5)
    # Position 1: excluded (NaN)
    # Position 2: valid (32, not NaN, mask=1, |32-0|>0.5)
    # Position 3: excluded (NaN + mask=0)
    # Position 4: excluded (mask=0)
    expected = torch.abs(preds[torch.tensor([0, 2])] - labels[torch.tensor([0, 2])]).mean()
    result = masked_mae(preds, labels, null_val, label_mask=label_mask)

    assert torch.isclose(result, expected, atol=ATOL), \
        f"NaN + label_mask combined: got {result:.4f}, expected {expected:.4f}"
    print("  PASS test_nan_plus_label_mask")


def test_all_nan_returns_zero():
    """If all labels are NaN, loss should be 0 (no valid positions)."""
    preds = torch.tensor([1.0, 2.0, 3.0])
    labels = torch.tensor([float("nan"), float("nan"), float("nan")])
    null_val = torch.tensor(0.0)

    result = masked_mae(preds, labels, null_val)
    assert result.item() == 0.0, f"All-NaN labels should give 0 loss, got {result:.6f}"
    print("  PASS test_all_nan_returns_zero")


def test_compute_all_metrics_consistent():
    """compute_all_metrics should return the same values as individual calls."""
    torch.manual_seed(99)
    preds = torch.randn(200) * 10 + 50
    labels = torch.randn(200) * 10 + 50
    labels[torch.rand(200) > 0.5] = float("nan")
    null_val = labels[~labels.isnan()].min()

    mae, mape, rmse = compute_all_metrics(preds, labels, null_val)
    assert abs(mae - masked_mae(preds, labels, null_val).item()) < ATOL
    assert abs(mape - masked_mape(preds, labels, null_val).item()) < ATOL
    assert abs(rmse - masked_rmse(preds, labels, null_val).item()) < ATOL
    print("  PASS test_compute_all_metrics_consistent")


def test_realistic_point_missing_scenario():
    """
    Simulates the actual training scenario: SD-like traffic data with 75%
    point-missing. Verifies that masked_mae reports the true MAE over valid
    positions, not a diluted value.
    """
    torch.manual_seed(7)
    B, H, N = 8, 12, 716

    labels = torch.randn(B, H, N) * 184.3 + 247.2
    preds = labels + torch.randn(B, H, N) * 5

    keep = torch.rand(B, H, N) > 0.75
    labels_masked = labels.clone()
    labels_masked[~keep] = float("nan")

    null_val = labels_masked[~labels_masked.isnan()].min()
    v = valid_mask(labels_masked, null_val)

    result = masked_mae(preds, labels_masked, null_val)
    expected = torch.abs(preds[v] - labels_masked[v]).mean()

    ratio = result / expected
    assert torch.isclose(result, expected, rtol=0.01), (
        f"Realistic scenario: result/expected ratio = {ratio:.4f} (should be ~1.0)\n"
        f"  Result:   {result:.4f}\n"
        f"  Expected: {expected:.4f}"
    )
    print(f"  PASS test_realistic_point_missing_scenario (ratio: {ratio:.4f})")


# ── main ─────────────────────────────────────────────────────────────────

def main():
    print("Testing masked metrics...\n")

    test_no_nan_no_nullval()
    test_null_val_excluded()
    test_null_val_tolerance()
    test_nan_labels_excluded_from_denominator()
    test_nan_labels_excluded_mse()
    test_nan_labels_excluded_mape()
    test_label_mask_further_restricts()
    test_nan_plus_label_mask()
    test_all_nan_returns_zero()
    test_compute_all_metrics_consistent()
    test_realistic_point_missing_scenario()

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
