import torch
import numpy as np
import scipy.sparse as sp

from src.base.engine import BaseEngine
from src.utils.metrics import masked_mape, masked_mae, masked_rmse, compute_all_metrics


class PDFormer_Engine(BaseEngine):
    def __init__(self, lape_dim=8, random_flip=True, set_loss='masked_mae', **args):
        super(PDFormer_Engine, self).__init__(**args)

        self.lape_dim = lape_dim
        self.random_flip = random_flip
        self.set_loss = set_loss

        self.lap_mx = self._cal_lape(self.model.adj_mx).to(self._device)

    def _calculate_normalized_laplacian(self, adj):
        adj = sp.coo_matrix(adj)
        d = np.array(adj.sum(1))
        isolated_point_num = np.sum(np.where(d, 0, 1))
        self._logger.info(f'Number of isolated points: {isolated_point_num}')

        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        return normalized_laplacian, isolated_point_num

    def _cal_lape(self, adj_mx):
        L, isolated_point_num = self._calculate_normalized_laplacian(adj_mx)
        eig_val, eig_vec = np.linalg.eig(L.toarray())
        idx = eig_val.argsort()
        eig_vec = np.real(eig_vec[:, idx])

        laplacian_pe = torch.from_numpy(
            eig_vec[:, isolated_point_num + 1:self.lape_dim + isolated_point_num + 1]
        ).float()
        laplacian_pe.requires_grad = False
        return laplacian_pe

    def _get_lap_mx(self, is_train=False):
        lap_mx = self.lap_mx
        if is_train and self.random_flip:
            sign_flip = torch.rand(lap_mx.size(1), device=lap_mx.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            lap_mx = lap_mx * sign_flip.unsqueeze(0)
        return lap_mx

    def forward(self, X, label, isTrain=False):
        lap_mx = self._get_lap_mx(is_train=isTrain)
        pred = self.model(X, lap_mx=lap_mx)
        return pred, label, None

    def train_batch(self):
        self.model.train()

        self.current_available_sensors = self._dataloader['train_loader'].available_sensors

        train_loss = []
        train_mape = []
        train_rmse = []

        for X, label, x_mask, label_mask in self._dataloader['train_loader'].get_iterator():
            self._optimizer.zero_grad()

            X, label = self._to_device(self._to_tensor([X, label]))
            self.current_x_mask = self._to_device(self._to_tensor(x_mask))
            self.current_label_mask = self._to_device(self._to_tensor(label_mask))

            pred, label, _ = self.forward(X, label, isTrain=True)

            loss = self.model.calculate_loss_without_predict(
                y_true=label,
                y_predicted=pred,
                batches_seen=self._iter_cnt,
                set_loss=self.set_loss
            )

            loss.backward()

            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)

            self._optimizer.step()

            pred, label = self._inverse_transform([pred, label])

            mask_value = self.mask_value(label)

            mape = masked_mape(pred, label, mask_value).item()
            rmse = masked_rmse(pred, label, mask_value).item()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

            self._iter_cnt += 1

            extra_train_logs = {}

        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse), extra_train_logs

    def evaluate(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)

        self.model.eval()

        preds = []
        labels = []

        with torch.no_grad():
            self.current_available_sensors = self._dataloader[mode + '_loader'].available_sensors

            for X, label, x_mask, label_mask in self._dataloader[mode + '_loader'].get_iterator():
                X, label = self._to_device(self._to_tensor([X, label]))
                self.current_x_mask = self._to_device(self._to_tensor(x_mask))
                self.current_label_mask = self._to_device(self._to_tensor(label_mask))

                pred, label, _ = self.forward(X, label, isTrain=False)
                pred, label = self._inverse_transform([pred, label])

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        mask_value = self.mask_value(labels)

        if mode == 'val':
            mae = masked_mae(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse

        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []

            for i in range(self.model.output_window):
                res = compute_all_metrics(preds[:, i, :], labels[:, i, :], mask_value)
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                self._wandb_logger.log_metrics({
                    f'test/horizon_{i+1}/mae': res[0],
                    f'test/horizon_{i+1}/mape': res[1],
                    f'test/horizon_{i+1}/rmse': res[2]
                }, step=self.epoch + 1)
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])

            self._logger.info(
                'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'.format(
                    np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)
                )
            )

            self._wandb_logger.log_metrics({
                'test/avg_mae': np.mean(test_mae),
                'test/avg_mape': np.mean(test_mape),
                'test/avg_rmse': np.mean(test_rmse)
            }, step=self.epoch + 1)

            training_available_sensors = self._dataloader['train_loader'].available_sensors
            if training_available_sensors is not None:
                avail = training_available_sensors.squeeze() == 1
                unavail = training_available_sensors.squeeze() == 0

                res = compute_all_metrics(preds[:, :, avail], labels[:, :, avail], mask_value)
                self._logger.info(
                    'Available Sensors - Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'.format(
                        res[0], res[2], res[1]
                    )
                )
                self._wandb_logger.log_metrics({
                    'test/available_sensors/avg_mae': res[0],
                    'test/available_sensors/avg_mape': res[1],
                    'test/available_sensors/avg_rmse': res[2]
                }, step=self.epoch + 1)

                res = compute_all_metrics(preds[:, :, unavail], labels[:, :, unavail], mask_value)
                self._logger.info(
                    'Unavailable Sensors - Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'.format(
                        res[0], res[2], res[1]
                    )
                )
                self._wandb_logger.log_metrics({
                    'test/unavailable_sensors/avg_mae': res[0],
                    'test/unavailable_sensors/avg_mape': res[1],
                    'test/unavailable_sensors/avg_rmse': res[2]
                }, step=self.epoch + 1)

            return np.mean(test_mae), np.mean(test_mape), np.mean(test_rmse)

        else:
            raise ValueError(f'Invalid mode {mode}')