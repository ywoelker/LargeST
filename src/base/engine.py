import os
import time
import torch
import numpy as np
from tqdm import tqdm
from src.utils.metrics import masked_mape, masked_mae
from src.utils.metrics import masked_rmse
from src.utils.metrics import compute_all_metrics
from src.utils.dataloader import DataLoader
from src.utils.logging import WandbLogger

class BaseEngine():
    def __init__(self, device, model, dataloader: dict[str,DataLoader], scaler, sampler, loss_fn, lrate, optimizer, \
                 scheduler, clip_grad_value, max_epochs, patience, log_dir, logger, seed, wandb_logger:WandbLogger):
        super().__init__()
        self._device = device
        self.model = model
        self.model.to(self._device)

        self._dataloader = dataloader
        self._scaler = scaler

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

        self.label_mask_value = self._scaler.transform(torch.tensor([0])).to(self._device)[0].to(torch.float)
        self._logger.info('The number of parameters: {}'.format(self.model.param_num())) 

        torch.manual_seed(seed)
        self.base_seeds = torch.randint(0, 10000, (self._max_epochs,)).tolist()


    def _to_device(self, tensors):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        else:
            return tensors.to(self._device)


    def _to_numpy(self, tensors):
        if isinstance(tensors, list):
            return [tensor.detach().cpu().numpy() for tensor in tensors]
        else:
            return tensors.detach().cpu().numpy()


    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [torch.tensor(array, dtype=torch.float32) for array in nparray]
        else:
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
        

    def forward(self, X, label, isTrain = False):
        # TODO: inverse transform of the label after the masking create mean values which are not masked.
        pred = self.model(X, label)
        return pred, label, None
    
 
    def loss(self, pred, label, mask_value, loss_container):
        loss = self._loss_fn(pred, label, mask_value, label_mask= self.current_label_mask)
        return loss
    
    def mask_value(self, label):
        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(0)
        if label.min() < 1:
            mask_value = label.min()
        
        if torch.isnan(label.min()).any():
            def nanmin(tensor):
                max_value = torch.finfo(tensor.dtype).max
                output = tensor.nan_to_num(max_value).min()
                return output
            
            mask_value_nanmin = nanmin(label)

            if mask_value_nanmin < 1:
                mask_value = mask_value_nanmin

        return mask_value

    def train_batch(self):
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader['train_loader'].shuffle()

        
        print('Check label mask value', self.label_mask_value)
        self.current_available_sensors = self._dataloader['train_loader'].available_sensors
        for X, label, x_mask, label_mask in tqdm(self._dataloader['train_loader'].get_iterator(),total = self._dataloader['train_loader'].num_batch, desc=f'Training - {train_loss[-1] if len(train_loss) > 0 else "N/A"}'):
            self._optimizer.zero_grad()

            # X (b, t, n, f), label (b, t, n, 1)
            X, label = self._to_device(self._to_tensor([X, label]))
            #TODO: The problem is that after the next line this has 9k entries `((self._inverse_transform([label])[0] > 0.0) & (self._inverse_transform([label])[0] < 0.1) ).sum()`
            # Before this line this is 0

            self.current_x_mask = self._to_device(self._to_tensor(x_mask))
            self.current_label_mask = self._to_device(self._to_tensor(label_mask))
            

            labels_as_input_to_model = torch.where(self.current_label_mask.to(bool), label, self.label_mask_value)

            pred, labels_as_input_to_model, loss_container = self.forward(X, labels_as_input_to_model, isTrain=True)            
            pred, label = self._inverse_transform([pred, label])
    
            mask_value = self.mask_value(label)

            if self._iter_cnt == 0:
                print('Check mask value', mask_value)

            loss = self.loss(pred, label, mask_value, loss_container)

            mape = masked_mape(pred, label, mask_value, label_mask= self.current_label_mask).item()
            rmse = masked_rmse(pred, label, mask_value, label_mask= self.current_label_mask).item()

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            train_loss.append(loss.item())
            train_mape.append(mape)
            train_rmse.append(rmse)

            self._iter_cnt += 1
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)


    def train(self):
        self._logger.info('Start training!')

        wait = 0
        min_loss = np.inf
        for self.epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mape, mtrain_rmse = self.train_batch()            
            t2 = time.time()
          
            # Log epoch-level training metrics
            if self._wandb_logger is not None:
                self._wandb_logger.log_metrics({
                    'train/loss': mtrain_loss,
                    'train/mape': mtrain_mape,
                    'train/rmse': mtrain_rmse,
                    'train/time_per_epoch': t2 - t1
                })

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
                })
                
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

        self.evaluate('test')


    def evaluate(self, mode) -> tuple:
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            self.current_available_sensors = self._dataloader[mode + '_loader'].available_sensors
            for batch_i, (X, label, x_mask, label_mask) in enumerate(self._dataloader[mode + '_loader'].get_iterator()):
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))

                self.current_x_mask = self._to_device(self._to_tensor(x_mask))
                self.current_label_mask = self._to_device(self._to_tensor(label_mask))
     
                pred, label, _ = self.forward(X, label, isTrain=False)
                pred, label = self._inverse_transform([pred, label])

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value = self.mask_value(labels)

        print('Check mask value for evaluation: ', mask_value)

        print((labels == mask_value).sum())

        if mode == 'val':
            mae = masked_mae(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return mae, mape, rmse

        elif mode == 'test':
            test_mae = []
            test_mape = []
            test_rmse = []
            print('Check mask value', mask_value)

            for i in range(self.model.horizon):
                res = compute_all_metrics(preds[:,i,:], labels[:,i,:], mask_value)
                log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                self._wandb_logger.log_metrics({
                    f'test/horizon_{i+1}/mae': res[0],
                    f'test/horizon_{i+1}/mape': res[1],
                    f'test/horizon_{i+1}/rmse': res[2]
                })
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])

            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            self._wandb_logger.log_metrics({
                'test/avg_mae': np.mean(test_mae),
                'test/avg_mape': np.mean(test_mape),
                'test/avg_rmse': np.mean(test_rmse)
            })
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))


            training_available_sensors = self._dataloader['train_loader'].available_sensors
            if training_available_sensors is not None:

                for i in range(self.model.horizon):
                    res = compute_all_metrics(preds[:,i,training_available_sensors.squeeze() == 1], labels[:,i,training_available_sensors.squeeze() == 1], mask_value)
                    log = '\tAvailable Sensors - Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                    self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                    self._wandb_logger.log_metrics({
                        f'test/available_sensors/horizon_{i+1}/mae': res[0],
                        f'test/available_sensors/horizon_{i+1}/mape': res[1],
                        f'test/available_sensors/horizon_{i+1}/rmse': res[2]
                    })

                res = compute_all_metrics(
                    preds[:, :, training_available_sensors.squeeze() == 1],
                    labels[:, :, training_available_sensors.squeeze() == 1],
                    mask_value
                )
                log = 'Available Sensors - Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(res[0], res[2], res[1]))
                self._wandb_logger.log_metrics({
                    'test/available_sensors/avg_mae': res[0],
                    'test/available_sensors/avg_mape': res[1],
                    'test/available_sensors/avg_rmse': res[2]
                })


                ## Unavailable sensors

                for i in range(self.model.horizon):
                    res = compute_all_metrics(preds[:,i,training_available_sensors.squeeze() == 0], labels[:,i,training_available_sensors.squeeze() == 0], mask_value)
                    log = '\tUnavailable Sensors - Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                    self._logger.info(log.format(i + 1, res[0], res[2], res[1]))
                    self._wandb_logger.log_metrics({
                        f'test/unavailable_sensors/horizon_{i+1}/mae': res[0],
                        f'test/unavailable_sensors/horizon_{i+1}/mape': res[1],
                        f'test/unavailable_sensors/horizon_{i+1}/rmse': res[2]
                    })

                res = compute_all_metrics(
                    preds[:, :, training_available_sensors.squeeze() == 0],
                    labels[:, :, training_available_sensors.squeeze() == 0],
                    mask_value
                )
                log = 'Unavailable Sensors - Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(res[0], res[2], res[1]))
                self._wandb_logger.log_metrics({
                    'test/unavailable_sensors/avg_mae': res[0],
                    'test/unavailable_sensors/avg_mape': res[1],
                    'test/unavailable_sensors/avg_rmse': res[2]
                })

            return np.mean(test_mae), np.mean(test_mape), np.mean(test_rmse)

        else:
            raise ValueError('Invalid mode {}'.format(mode))