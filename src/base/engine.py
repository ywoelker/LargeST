import os
import time
import torch
import numpy as np
from tqdm import tqdm
from wandb.jupyter import logger
from src.utils.metrics import masked_mape, masked_mae
from src.utils.metrics import masked_rmse
from src.utils.metrics import compute_all_metrics
from src.utils.dataloader import DataLoader, get_dataset_info, load_adj_from_numpy
from src.utils.logging import WandbLogger
from collections import defaultdict
import matplotlib.pyplot as plt


class BaseEngine():
    def __init__(self, device, model, dataloader: dict[str,DataLoader], scaler, sampler, loss_fn, lrate, optimizer, \
                 scheduler, clip_grad_value, max_epochs, patience, log_dir, logger, seed, wandb_logger:WandbLogger, training_timeout_min:int, trainable_model: bool = True):
        super().__init__()
        self._device = device
        self.model = model
        self.model.to(self._device)

        self._dataloader = dataloader
        self._scaler = scaler
        self._scaler.to(self._device)

        self._loss_fn = loss_fn
        self._lrate = lrate
        self._optimizer = optimizer
        self._lr_scheduler = scheduler
        self._clip_grad_value = clip_grad_value

        self._max_epochs = max_epochs
        self._patience = patience
        self._iter_cnt = 0
        self._save_path = log_dir
        self._logger = logger
        self._seed = seed
        self._wandb_logger = wandb_logger
        self._training_timeout_min = training_timeout_min
        self.trainable_model = trainable_model 

        self.label_mask_value = self._scaler.transform(torch.tensor([0.0]).to(dtype = torch.float32, device = self._device))[0]
        self._logger.info('The number of parameters: {}'.format(self.model.param_num())) 

        torch.manual_seed(seed)
        self.base_seeds = torch.randint(0, 10000, (self._max_epochs,)).tolist()


    def _to_device(self, tensors):
        non_blocking = self._device.type == 'cuda'
        if isinstance(tensors, list):
            return [tensor.to(self._device, non_blocking=non_blocking) for tensor in tensors]
        else:
            return tensors.to(self._device, non_blocking=non_blocking)


    def _to_numpy(self, tensors):
        if isinstance(tensors, list):
            return [tensor.detach().cpu().numpy() for tensor in tensors]
        else:
            return tensors.detach().cpu().numpy()


    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [self._to_tensor(array) for array in nparray]
        if torch.is_tensor(nparray):
            # The dataloader already hands out device tensors; re-wrapping them with
            # torch.tensor() would warn and copy back through the host.
            return nparray if nparray.dtype == torch.float32 else nparray.float()
        return torch.tensor(nparray, dtype=torch.float32)


    def _inverse_transform(self, tensors):
        def inv(tensor):
            return self._scaler.inverse_transform(tensor)

        if isinstance(tensors, list):
            return [inv(tensor) for tensor in tensors]
        else:
            return inv(tensors)


    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'final_model_s{}.pt'.format(self._seed)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))


    def load_model(self, save_path):
        filename = 'final_model_s{}.pt'.format(self._seed)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))   
        

    def forward(self, X, label, isTrain = False, query_node = None):
        # TODO: inverse transform of the label after the masking create mean values which are not masked.
        pred = self.model(X, label)
        
        if query_node is not None:
            pred = pred[:, :, query_node, :]
            label = label[:, :, query_node, :]
        
        return pred, label, None
    
 
    def loss(self, pred, label, mask_value, loss_container):
        loss = self._loss_fn(pred, label, mask_value, label_mask= self.current_label_mask)
        return loss
    

    def extra_loss_and_logs(self, pred, label, mask_value, loss_container, epoch):
        """
        Returns:
            extra_loss (torch.Tensor): scalar tensor to add to total loss
            logs (dict): python floats or 0-d tensors for logging
        """
        return torch.tensor(0.0, device=self._device), {}
    

    def log_custom_visuals(self, epoch):
        pass


    
    def mask_value(self, label):
        if torch.isnan(label).any():
            # NaN labels come from output masking. The smallest valid label
            # is inverse_transform(label_mask_value) which is exactly 0, but
            # float32 round-trip can land on a tiny negative (device-dependent).
            # Use 0 directly to avoid CPU/MPS divergence.
            return torch.tensor(0.0)

        mask_value = torch.tensor(0.0)
        if label.min() < 1:
            mask_value = label.min()
        return mask_value

    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        extra_logs_acc = defaultdict(list)
        self._dataloader['train_loader'].shuffle()

        
        print('Check label mask value', self.label_mask_value)
        self.current_available_sensors = self._dataloader['train_loader'].available_sensors
        for X, label, x_mask, label_mask in tqdm(self._dataloader['train_loader'].get_iterator(),total = self._dataloader['train_loader'].num_batch, desc=f'Training - {train_loss[-1] if len(train_loss) > 0 else "N/A"}'):
            self._optimizer.zero_grad()

            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))
            
            self.current_x_mask = self._to_device(self._to_tensor(x_mask))
            self.current_label_mask = self._to_device(self._to_tensor(label_mask))


            labels_as_input_to_model = torch.where(self.current_label_mask.to(bool), label, self.label_mask_value)

            pred, labels_as_input_to_model, loss_container = self.forward(X, labels_as_input_to_model, isTrain=True)
            pred, label = self._inverse_transform([pred, label])

            mask_value = self.mask_value(label)

            if self._iter_cnt == 0:
                print('Check mask value', mask_value)

            loss = self.loss(pred, label, mask_value, loss_container)

            extra_loss, extra_logs = self.extra_loss_and_logs(pred, label, mask_value, loss_container, self.epoch)


            mape = masked_mape(pred, label, mask_value, label_mask= self.current_label_mask).item()
            rmse = masked_rmse(pred, label, mask_value, label_mask= self.current_label_mask).item()

            loss_total = loss + extra_loss

            loss_total.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss_total.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

            # accumulate extra logs
            for k, v in extra_logs.items():
                extra_logs_acc[k].append(float(v))

            self._iter_cnt += 1
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse), extra_logs_acc


    def train(self):
        self._logger.info('Start training!')

        wait = 0
        min_loss = np.inf

        t0 = time.time()

        for self.epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_rmse, train_extra_logs = self.train_batch()            
            t2 = time.time()
          
            # Log epoch-level training metrics
            if self._wandb_logger is not None:
                train_log = {
                'train/loss': mtrain_loss,
                'train/mape': mtrain_mape,
                'train/rmse': mtrain_rmse,
                'train/time_per_epoch': t2 - t1,
                }
                for k, v_list in train_extra_logs.items():
                    train_log[f'train/{k}'] = float(np.mean(v_list))
                self._wandb_logger.log_metrics(train_log, step=self.epoch+1)


            v1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = self.evaluate('val')
            v2 = time.time()


            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            # Log epoch-level validation metrics
            if self._wandb_logger is not None:
                self._wandb_logger.log_metrics({
                    'val/loss': mvalid_loss,
                    'val/mape': mvalid_mape,
                    'val/rmse': mvalid_rmse,
                    'val/time': v2 - v1,
                    'lr': cur_lr,
                    'epoch': self.epoch + 1
                }, step=self.epoch+1)
                
            message = 'Epoch: {:03d}, Train Loss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Valid Loss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, LR: {:.4e}'
            self._logger.info(message.format(self.epoch + 1, mtrain_loss, mtrain_rmse, mtrain_mape, \
                                             mvalid_loss, mvalid_rmse, mvalid_mape, \
                                             (t2 - t1), (v2 - v1), cur_lr))

            if mvalid_loss < min_loss:
                self.save_model(self._save_path)
                self._logger.info('Val loss decrease from {:.4f} to {:.4f}'.format(min_loss, mvalid_loss))
                min_loss = mvalid_loss
                wait = 0
                self.best_epoch = self.epoch
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info('Early stop at epoch {} with best epoch being {}, loss = {:.6f}'.format(self.epoch + 1, self.best_epoch + 1, min_loss))
                    break

            t_timeout = time.time()

            if self._training_timeout_min is not None and self._training_timeout_min > 0 and t_timeout - t0 > self._training_timeout_min * 60:
                self._logger.info('Timeout reached. Ending training at epoch {} with best epoch being {}, loss = {:.6f}'.format(self.epoch + 1, self.best_epoch + 1, min_loss))
                break

        self.evaluate('test')
        
        if 'test_loader_drop' in self._dataloader and not self._dataloader['test_loader_drop'].drop_unavailable_sensors:
            self.benchmark_interpolation_for_dropped_sensors()
        benchmark_results = self.benchmark_inference_time(self._dataloader['benchmark_loader'])
        
        self._logger.info('Benchmark inference time: {:.4f}s +- {:.4f}s'.format(benchmark_results[0], benchmark_results[1]))
        


    def evaluate(self, mode) -> tuple:
        if mode == 'test' and self.trainable_model:
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        label_masks = []
        with torch.no_grad():
            self.current_available_sensors = self._dataloader[mode + '_loader'].available_sensors
            for X, label, x_mask, label_mask in tqdm(self._dataloader[mode + '_loader'].get_iterator(),total = self._dataloader[mode + '_loader'].num_batch, desc=f'{"Validation" if mode == "val" else "Test"}'):
                
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))

                self.current_x_mask = self._to_device(self._to_tensor(x_mask))
                self.current_label_mask = self._to_device(self._to_tensor(label_mask))
     
                pred, label, _ = self.forward(X, label, isTrain=False)
                pred, label = self._inverse_transform([pred, label])

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())
                label_masks.append(label_mask.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        label_masks = torch.cat(label_masks, dim = 0)

        # handle the precision issue when performing inverse transform to label
        mask_value = self.mask_value(labels)

        print('Check mask value for evaluation: ', mask_value)

        print((labels == mask_value).sum())

        if mode == 'val':
            mae, mape, rmse = compute_all_metrics(preds, labels, mask_value, label_mask = label_masks)
            
            #### IMPORTANT: The validation set masks out the labels for the unseen sensors, so it makes no sense to calculate the metrics for the unseen sensors here in any kind.             
            
            return mae, mape, rmse

        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            print('Check mask value', mask_value)

            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:,i,:], labels[:,i,:], mask_value, label_mask = label_masks[:,i,:])
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                self._wandb_logger.log_metrics({
                    f'test/horizon_{i+1}/mae': res[0],
                    f'test/horizon_{i+1}/mape': res[1],
                    f'test/horizon_{i+1}/rmse': res[2]
                }, step=self.epoch+1)
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])

            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            self._wandb_logger.log_metrics({
                'test/avg_mae': np.mean(test_mae),
                'test/avg_mape': np.mean(test_mape),
                'test/avg_rmse': np.mean(test_rmse)
            }, step=self.epoch+1)
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))


            training_available_sensors = self._dataloader['train_loader'].available_sensors
            if training_available_sensors is not None:

                for i in range(self.model.horizon):
                    res = compute_all_metrics(preds[:,i,training_available_sensors.squeeze() == 1], labels[:,i,training_available_sensors.squeeze() == 1], mask_value, label_mask = label_masks[:,i,training_available_sensors.squeeze() == 1])
                    log = '\tAvailable Sensors - Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                    self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                    self._wandb_logger.log_metrics({
                        f'test/available_sensors/horizon_{i+1}/mae': res[0],
                        f'test/available_sensors/horizon_{i+1}/mape': res[1],
                        f'test/available_sensors/horizon_{i+1}/rmse': res[2]
                    }, step=self.epoch+1)

                res = compute_all_metrics(
                    preds[:, :, training_available_sensors.squeeze() == 1],
                    labels[:, :, training_available_sensors.squeeze() == 1],
                    mask_value,
                    label_mask = label_masks[:, :, training_available_sensors.squeeze() == 1]
                )
                log = 'Available Sensors - Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(res[0], res[2], res[1]))
                self._wandb_logger.log_metrics({
                    'test/available_sensors/avg_mae': res[0],
                    'test/available_sensors/avg_mape': res[1],
                    'test/available_sensors/avg_rmse': res[2]
                }, step=self.epoch+1)


                ## Unavailable sensors

                for i in range(self.model.horizon):
                    res = compute_all_metrics(preds[:,i,training_available_sensors.squeeze() == 0], labels[:,i,training_available_sensors.squeeze() == 0], mask_value, label_mask = label_masks[:,i,training_available_sensors.squeeze() == 0])
                    log = '\tUnavailable Sensors - Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                    self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                    self._wandb_logger.log_metrics({
                        f'test/unavailable_sensors/horizon_{i+1}/mae': res[0],
                        f'test/unavailable_sensors/horizon_{i+1}/mape': res[1],
                        f'test/unavailable_sensors/horizon_{i+1}/rmse': res[2]
                    }, step=self.epoch+1)

                res = compute_all_metrics(
                    preds[:, :, training_available_sensors.squeeze() == 0],
                    labels[:, :, training_available_sensors.squeeze() == 0],
                    mask_value, label_mask = label_masks[:, :, training_available_sensors.squeeze() == 0]
                )
                log = 'Unavailable Sensors - Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(res[0], res[2], res[1]))
                self._wandb_logger.log_metrics({
                    'test/unavailable_sensors/avg_mae': res[0],
                    'test/unavailable_sensors/avg_mape': res[1],
                    'test/unavailable_sensors/avg_rmse': res[2]
                }, step=self.epoch+1)

            return np.mean(test_mae), np.mean(test_mape), np.mean(test_rmse)

        else:
            raise ValueError('Invalid mode {}'.format(mode))
        
        
        
    def benchmark_interpolation_for_dropped_sensors(self) -> tuple:
        mode = 'test'
        self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        label_masks = []
        self.current_available_sensors = self._dataloader[mode + '_loader_drop'].available_sensors
        
        path, adj_path, _ = get_dataset_info(self._dataloader['test_loader_drop'].dataset)
        
        import pandas as pd
        meta_data = pd.read_csv(path + f'/{self._dataloader["test_loader_drop"].dataset.lower()}_meta.csv')
        from sklearn.metrics import pairwise_distances
        
        distances = pairwise_distances(meta_data[['Lat', 'Lng']], metric='euclidean')
        distances = distances #+ np.diag(np.inf * np.ones(distances.shape[0]))
        distances[:, self.current_available_sensors.squeeze() == 0] = np.inf

        K = 5
        top_k_indices = np.argsort(distances, axis=1)[:,:K]
        
        # print('Top-K indices for each sensor: ', top_k_indices)
        
        # adj_mx = load_adj_from_numpy(adj_path)
        
        # # set diagonal of adjacency matrix to 0
        # adj_mx = adj_mx - np.diag(np.diag(adj_mx))
        
        # get the TOP-K indices of the columns for each row with the largest values in the adjacency matrix
        # top_k_indices = np.argsort(adj_mx, axis=1)[:, -K:]
        
        with torch.no_grad():
            for (X, label, x_mask, label_mask), (X_test, label_test,_, label_mask_test) in tqdm(
                zip(
                    self._dataloader[mode + '_loader_drop'].get_iterator(),
                    self._dataloader[mode + '_loader'].get_iterator(),
                ),
                    total = self._dataloader[mode + '_loader_drop'].num_batch, desc=f'{"Validation" if mode == "val" else "Test"}'):
                
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                X_test, label_test = self._to_device(self._to_tensor([X_test, label_test]))

                self.current_x_mask = self._to_device(self._to_tensor(x_mask))
                self.current_label_mask = self._to_device(self._to_tensor(label_mask_test))
        
                pred, label, _ = self.forward(X, label, isTrain=False)
                ####
                ## for those sensors that arent available take the average of the top-K neighbors to fill in the missing values
                
                pred[:, :, self.current_available_sensors.squeeze() == 0, :] = torch.nan
                
                # for nb in range(pred.shape[0]):
                for j in range(pred.shape[2]):
                    if not self.current_available_sensors.squeeze()[j]:
                        # for t in range(pred.shape[1]):    
                            # assert np.isin(top_k_indices[j],np.argwhere( self.current_available_sensors).squeeze()).all(), "Top-K indices should only include available sensors"
                            # print(f'Replacing sensor {j} with sensors {top_k_indices[j]}: Absolute difference: {torch.mean(torch.abs(pred[:, t, top_k_indices[j][0], :] - pred[:, t, j, :]))}')
                            replacement_mean = torch.mean(pred[:, :, top_k_indices[j], :], dim=2)
                            
                            assert pred[:, :, j, :].shape == replacement_mean.shape, "Shape mismatch between pred and replacement mean"
                            
                            pred[:, :, j, :] = replacement_mean
                ####

                assert not torch.isnan(pred).any(), "Predictions contain NaN values after interpolation"                
                
                pred, label = self._inverse_transform([pred, label])
                label_test = self._inverse_transform(label_test)
                
                
                # print(torch.mean(torch.abs(label_test[:, :, self.current_available_sensors.squeeze() == 1] - label[:, :, self.current_available_sensors.squeeze() == 1])))

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label_test.squeeze(-1).cpu())
                label_masks.append(self.current_label_mask.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        label_masks = torch.cat(label_masks, dim = 0)

        # handle the precision issue when performing inverse transform to label
        mask_value = self.mask_value(labels)

        print('Check mask value for evaluation: ', mask_value)

        print((labels == mask_value).sum())
        
        
        np.savez(
            self._save_path + '/results_interpolation.npz',
            preds=preds.numpy(),
            labels=labels.numpy(),
            label_masks=label_masks.numpy(),
            top_k_indices=top_k_indices,
            mask_value=mask_value,
            training_available_sensors=self.current_available_sensors
        )

        test_mae = []
        test_mape = []
        test_rmse = []
        print('Check mask value', mask_value)

        for i in range(self.model.horizon):
            res = compute_all_metrics(preds[:,i,:], labels[:,i,:], mask_value, label_mask = label_masks[:,i,:])
            log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
            self._wandb_logger.log_metrics({
                f'test_dropped/horizon_{i+1}/mae': res[0],
                f'test_dropped/horizon_{i+1}/mape': res[1],
                f'test_dropped/horizon_{i+1}/rmse': res[2]
            }, step=self.epoch+1)
            test_mae.append(res[0])
            test_mape.append(res[1])
            test_rmse.append(res[2])

        log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
        self._wandb_logger.log_metrics({
            'test_dropped/avg_mae': np.mean(test_mae),
            'test_dropped/avg_mape': np.mean(test_mape),
            'test_dropped/avg_rmse': np.mean(test_rmse)
        }, step=self.epoch+1)
        self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))


        training_available_sensors = self._dataloader['train_loader'].available_sensors
        if training_available_sensors is not None:

            for i in range(self.model.horizon):
                
                res = compute_all_metrics(preds[:,i,training_available_sensors.squeeze() == 1], labels[:,i,training_available_sensors.squeeze() == 1], mask_value, label_mask = label_masks[:,i,training_available_sensors.squeeze() == 1])
                log = '\tAvailable Sensors - Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                self._wandb_logger.log_metrics({
                    f'test_dropped/available_sensors/horizon_{i+1}/mae': res[0],
                    f'test_dropped/available_sensors/horizon_{i+1}/mape': res[1],
                    f'test_dropped/available_sensors/horizon_{i+1}/rmse': res[2]
                }, step=self.epoch+1)

            res = compute_all_metrics(
                preds[:, :, training_available_sensors.squeeze() == 1],
                labels[:, :, training_available_sensors.squeeze() == 1],
                mask_value,
                label_mask = label_masks[:, :, training_available_sensors.squeeze() == 1]   
            )
            log = 'Available Sensors - Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            self._logger.info(log.format(res[0], res[2], res[1]))
            self._wandb_logger.log_metrics({
                'test_dropped/available_sensors/avg_mae': res[0],
                'test_dropped/available_sensors/avg_mape': res[1],
                'test_dropped/available_sensors/avg_rmse': res[2]
            }, step=self.epoch+1)


            ## Unavailable sensors
            
                
            
                

            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:,i,training_available_sensors.squeeze() == 0], labels[:,i,training_available_sensors.squeeze() == 0], mask_value, label_mask = label_masks[:,i,training_available_sensors.squeeze() == 0])
                log = '\tUnavailable Sensors - Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                self._wandb_logger.log_metrics({
                    f'test_dropped/unavailable_sensors/horizon_{i+1}/mae': res[0],
                    f'test_dropped/unavailable_sensors/horizon_{i+1}/mape': res[1],
                    f'test_dropped/unavailable_sensors/horizon_{i+1}/rmse': res[2]
                }, step=self.epoch+1)

            res = compute_all_metrics(
                preds[:, :, training_available_sensors.squeeze() == 0],
                labels[:, :, training_available_sensors.squeeze() == 0],
                mask_value, label_mask = label_masks[:, :, training_available_sensors.squeeze() == 0]
            )
            log = 'Unavailable Sensors - Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            self._logger.info(log.format(res[0], res[2], res[1]))
            self._wandb_logger.log_metrics({
                'test_dropped/unavailable_sensors/avg_mae': res[0],
                'test_dropped/unavailable_sensors/avg_mape': res[1],
                'test_dropped/unavailable_sensors/avg_rmse': res[2]
            }, step=self.epoch+1)

            return np.mean(test_mae), np.mean(test_mape), np.mean(test_rmse)

        else:
            raise ValueError('Invalid mode {}'.format(mode))
        
        
    def benchmark_inference_time(self, data_loader) -> tuple:
        
        self.load_model(self._save_path)
        self.model.eval()
        
        # assert data_loader.bs == 1, "For benchmarking inference time, please set batch size to 1 to get more accurate measurement."
        
        query_nodes = np.random.randint(0, data_loader.n_sensors, (data_loader.num_batch,))

        with torch.no_grad():
            self.current_available_sensors = data_loader.available_sensors
            inference_time_numbers = [] 
            for batch_i, (X, label, x_mask, label_mask) in enumerate(data_loader.get_iterator()):
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                self.current_x_mask = self._to_device(self._to_tensor(x_mask))
                self.current_label_mask = self._to_device(self._to_tensor(label_mask))
     
     
                v1 = time.time()
                pred, label, _ = self.forward(X, label, isTrain=False, query_node=query_nodes[batch_i])
                
                v2 = time.time()
                inference_time_numbers.append(v2 - v1)
                pred, label = self._inverse_transform([pred, label])

        self._wandb_logger.log_metrics({
            f'benchmark/inference_time_mean': np.mean(inference_time_numbers),
            f'benchmark/inference_time_std': np.std(inference_time_numbers)
        }, step=self.epoch+1)
        
        return np.mean(inference_time_numbers), np.std(inference_time_numbers)