import torch
import numpy as np
from tqdm import tqdm
from src.base.engine import BaseEngine
from src.utils.metrics import masked_mape, masked_rmse, compute_all_metrics


class GMAN_Engine(BaseEngine):
    def __init__(self, **args):
        super(GMAN_Engine, self).__init__(**args)

    # Helper: build TE (temporal embeddings) from X
    def build_te(self, X, horizon, T=288):
        """
        Build GMAN temporal embedding tensor.
        X: [B, num_his, N, F]  (F includes tod, dow at indices 1:3)
        Return TE: [B, num_his + horizon, 2] with (dow, tod)
        """
        # B: batch size, H: num historical steps, N: num nodes, F: num features
        B, H, N, F = X.shape
        # Take time-of-day and day-of-week from first node (same for all nodes)
        # We turn these to integers so GMAN's temporal embedding works properly by one-hot encoding
        tod = (X[:, :, 0, 1] * (T - 1) + 0.5).long().clamp(0, T - 1)
        dow = (X[:, :, 0, 2] * 7 + 0.5).long() % 7

        # Predict future time slots
        steps = torch.arange(1, horizon + 1, device=X.device).unsqueeze(0)
        tod_f = (tod[:, -1:].expand(-1, horizon) + steps) % T
        carry = ((tod[:, -1:].expand(-1, horizon) + steps) // T)
        dow_f = (dow[:, -1:].expand(-1, horizon) + carry) % 7

        tod_all = torch.cat([tod, tod_f], dim=1)
        dow_all = torch.cat([dow, dow_f], dim=1)
        return torch.stack([dow_all, tod_all], dim=-1)  # [B, his+horizon, 2]

    # Training loop for one epoch
    def train_batch(self):
        self.model.train()
        train_loss, train_mape, train_rmse = [], [], []

        self._dataloader['train_loader'].shuffle()

        for X, label in tqdm(self._dataloader['train_loader'].get_iterator(),
                             total=self._dataloader['train_loader'].num_batch,
                             desc=f'Training - {train_loss[-1] if len(train_loss) > 0 else "N/A"}'):

            X, label = self._to_device(self._to_tensor([X, label]))
            self._optimizer.zero_grad()

            # Extract main input feature
            X_input = X[..., 0]  # [B, his, N]
            TE = self.build_te(X, horizon=self.model.horizon)

            # Forward
            pred = self.model(X_input, TE)
            pred, label = self._inverse_transform([pred, label])
            label = label.squeeze(-1)   # now pred,label -> [B, H, N]

            # Compute masked loss
            mask_value = self.mask_value(label)

            if self._iter_cnt == 0:
                print('Check mask value', mask_value)

            loss = self._loss_fn(pred, label, mask_value)
            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            # Backpropagation
            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

            self._iter_cnt += 1

        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)

    # Evaluation loop
    def evaluate(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds, labels = [], []

        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():
                X, label = self._to_device(self._to_tensor([X, label]))
                X_input = X[..., 0]
                TE = self.build_te(X, horizon=self.model.horizon)
                pred = self.model(X_input, TE)
                pred, label = self._inverse_transform([pred, label])
                label = label.squeeze(-1)   # now pred,label -> [B, H, N]

                preds.append(pred.cpu())
                labels.append(label.cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        mask_value = self.mask_value(labels)

        print('Check mask value for evaluation: ', mask_value)

        print((labels == mask_value).sum())


        if mode == 'val':
            mae = self._loss_fn(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse

        elif mode == 'test':
            test_mae, test_mape, test_rmse = [], [], []
            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:, i, :], labels[:, i, :], mask_value)
                self._logger.info(
                    f"Horizon {i+1}: MAE {res[0]:.4f}, RMSE {res[2]:.4f}, MAPE {res[1]:.4f}"
                )
                self._wandb_logger.log_metrics({
                    f'test/horizon_{i+1}/mae': res[0],
                    f'test/horizon_{i+1}/mape': res[1],
                    f'test/horizon_{i+1}/rmse': res[2]
                })
                test_mae.append(res[0]); test_mape.append(res[1]); test_rmse.append(res[2])

            self._logger.info(
                f"Average Test MAE: {np.mean(test_mae):.4f}, "
                f"RMSE: {np.mean(test_rmse):.4f}, "
                f"MAPE: {np.mean(test_mape):.4f}"
            )
            self._wandb_logger.log_metrics({
                'test/avg_mae': np.mean(test_mae),
                'test/avg_mape': np.mean(test_mape),
                'test/avg_rmse': np.mean(test_rmse)
            })