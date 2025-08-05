import torch
import numpy as np
from tqdm import tqdm
from src.base.engine import BaseEngine
from src.utils.metrics import masked_mape, masked_rmse
from src.utils.metrics import compute_all_metrics

from torch.profiler import profile, record_function, ProfilerActivity


class DeepStateGNN_Engine(BaseEngine):
    def __init__(self, **args):
        super(DeepStateGNN_Engine, self).__init__(**args)


    def train_batch(self):

        # prof = profile(
        #     activities=[
        #         ProfilerActivity.CPU,
        #         ProfilerActivity.CUDA],
        #     record_shapes=True,
        #     with_stack=True,
        #     profile_memory=True,
        #     with_flops=True
        # )
        self.model.train()

        train_loss = []
        train_mape = []
        train_rmse = []
        self._dataloader['train_loader'].shuffle()
        def trace_handler(prof):
            print(prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=5))
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA
            ], 
            record_shapes=True,
            with_stack=True,
            schedule=torch.profiler.schedule(
                        wait=5,
                        warmup=10,
                        active=2,
                        repeat=1),
            on_trace_ready=trace_handler
                        ) as prof:
            prof.stop()
            for X, label in tqdm(self._dataloader['train_loader'].get_iterator(),total = self._dataloader['train_loader'].num_batch, desc=f'Training - {train_loss[-1] if len(train_loss) > 0 else "N/A"}'):
                self._optimizer.zero_grad()


                X, label = self._to_device(self._to_tensor([X, label]))
                with record_function("model_inference"):
                    pred = self.model(X, label)
                    pred = torch.stack(pred, dim = 0)

                    pred, label = self._inverse_transform([pred, label])

                    # handle the precision issue when performing inverse transform to label
                    mask_value = torch.tensor(0)
                    if label.min() < 1:
                        mask_value = label.min()
                    if self._iter_cnt == 0:
                        print('check mask value', mask_value)
                with record_function("loss_computation"):
                    loss = self._loss_fn(pred, label, mask_value)
                    mape = masked_mape(pred, label, mask_value).item()
                    rmse = masked_rmse(pred, label, mask_value).item()

                    loss.backward()
                    if self._clip_grad_value != 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
                    self._optimizer.step()

                train_loss.append(loss.item())
                train_mape.append(mape)
                train_rmse.append(rmse)

                self._iter_cnt += 1

                prof.step()
        return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)
    
    def evaluate(self, mode):
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            for X, label in self._dataloader[mode + '_loader'].get_iterator():
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X, label)
                pred = torch.stack(pred, dim = 0)
                pred, label = self._inverse_transform([pred, label])

                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # handle the precision issue when performing inverse transform to label
        mask_value = torch.tensor(0)
        if labels.min() < 1:
            mask_value = labels.min()

        if mode == 'val':
            mae = self._loss_fn(preds, labels, mask_value).item()
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
                test_mae.append(res[0])
                test_mape.append(res[1])
                test_rmse.append(res[2])

            log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
            self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))


import torch
import numpy as np
from src.utils.metrics import masked_mae

def spatial_loss(node_vec1, node_vec2, supports, edge_indices):
    B = node_vec1.size(0)
    node_vec1 = node_vec1.permute(1, 0, 2, 3) # [N, B, 1, r]
    node_vec2 = node_vec2.permute(1, 0, 2, 3) # [N, B, 1, r]
    
    node_vec1_end, node_vec2_start = node_vec1[edge_indices[:, 0]], node_vec2[edge_indices[:, 1]] # [E, B, 1, r]
    attn1 = torch.einsum("ebhm,ebhm->ebh", node_vec1_end, node_vec2_start) # [E, B, 1]
    attn1 = attn1.permute(1, 0, 2) # [B, E, 1]

    one_matrix = torch.ones([node_vec2.shape[0]]).to(node_vec1.device)
    node_vec2_sum = torch.einsum("nbhm,n->bhm", node_vec2, one_matrix)
    attn_norm = torch.einsum("nbhm,bhm->nbh", node_vec1, node_vec2_sum)
    
    attn2 = attn_norm[edge_indices[:, 0]]  # [E, B, 1]
    attn2 = attn2.permute(1, 0, 2) # [B, E, 1]
    attn_score = attn1 / attn2 # [B, E, 1]
    
    d_norm = supports[0][edge_indices[:, 0], edge_indices[:, 1]]
    d_norm = d_norm.reshape(1, -1, 1).repeat(B, 1, attn_score.shape[-1])
    spatial_loss = torch.mean(attn_score.log() * d_norm)
    
    return spatial_loss

def bigst_loss(prediction, target, node_vec1, node_vec2, supports, use_spatial, mask_value):
    if use_spatial:
        supports = [support.to(prediction.device) for support in supports]
        edge_indices = torch.nonzero(supports[0] > 0)
        s_loss = spatial_loss(node_vec1, node_vec2, supports, edge_indices)
        return masked_mae(prediction, target, mask_value) - 0.3 * s_loss # 源代码：pipline.py line30
    else:
        return masked_mae(prediction, target, mask_value)