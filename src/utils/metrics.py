import torch

def _label_mask(label, null_val, atol=0.5):
    # print(null_val, torch.isnan(label).sum())
    if torch.isnan(null_val):
        mask = ~torch.isnan(label)
    else:
        mask = torch.abs(label - null_val) > atol

    # print(f"Mask has sum of {mask.sum().item()} and {(mask > 0).sum().item()} valid entries out of {mask.numel()} total entries.")
    if torch.isnan(label).any():
        nan_mask = ~torch.isnan(label)
        mask = mask & nan_mask

    mask = mask.float()

    # mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    # print(f"Mask has sum of {mask.sum().item()} and {(mask > 0).sum().item()} valid entries out of {mask.numel()} total entries.")
    return mask

def masked_mse(preds, labels, null_val, label_mask = None):
    mask = _label_mask(labels, null_val)
    if label_mask is not None:
        mask = mask * label_mask
    loss = (preds - labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sum(loss) / mask.sum().clamp(min=1)


def masked_rmse(preds, labels, null_val, label_mask = None):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val, label_mask= label_mask))


def masked_mae(preds, labels, null_val, label_mask = None):
    mask = _label_mask(labels, null_val)
    if label_mask is not None:
        mask = mask * label_mask
    loss = torch.abs(preds - labels)
    loss = loss #* mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sum(loss) / mask.sum().clamp(min=1)


def masked_mape(preds, labels, null_val, label_mask = None):
    mask = _label_mask(labels, null_val)
    if label_mask is not None:
        mask = mask * label_mask
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sum(loss) / mask.sum().clamp(min=1)


def compute_all_metrics(preds, labels, null_val):
    mae = masked_mae(preds, labels, null_val).item()
    mape = masked_mape(preds, labels, null_val).item()
    rmse = masked_rmse(preds, labels, null_val).item()
    return mae, mape, rmse