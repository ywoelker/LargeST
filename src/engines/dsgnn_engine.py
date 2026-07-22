import torch
import numpy as np
from tqdm import tqdm
from src.base.engine import BaseEngine
import matplotlib.pyplot as plt
from src.utils.metrics import masked_mape, masked_mae
from src.utils.metrics import masked_rmse
from src.utils.metrics import compute_all_metrics


class DSGNN_Engine(BaseEngine):
    def __init__(self, static_prefilter, additional_loss_weight, dsn_div_weight, dsn_div_margin, dsn_div_top_k, **args):
        super(DSGNN_Engine, self).__init__(**args)

        self.static_prefilter = static_prefilter
        self.additional_loss_weight = additional_loss_weight
        self.dsn_div_weight = dsn_div_weight
        self.dsn_div_margin = dsn_div_margin
        self.dsn_div_top_k = dsn_div_top_k
        self.embedding_evolution = {}

    def forward(self, X, label, isTrain = False, **kwargs):

        if self.static_prefilter is not None:
            static_assignment = self.static_prefilter
            
            if self.current_available_sensors is not None:
                static_assignment = static_assignment[self.current_available_sensors.squeeze() == 1]
            
        else:
            static_assignment = None

        if kwargs.get('query_node', None) is not None:
            query_node = kwargs['query_node']
            pred_dict = self.model(X.transpose(1,2), label, static_prefilter = static_assignment, query_index = query_node)
        else: 
            pred_dict = self.model(X.transpose(1,2), label, static_prefilter = static_assignment)
        pred = pred_dict['prediction']


        return pred, label, pred_dict
    

    def _take_first(self, a):
        if isinstance(a, np.ndarray):
            return a[:1]
        if torch.is_tensor(a):
            return a[:1].detach().cpu()
        try:
            return a[:1]
        except Exception:
            return a

    def cosine_repulsion_loss(self, Z, margin=0.3, eps=1e-8, top_k=False, k=None):
        """
        Compute the cosine repulsion loss for the given embeddings Z. That is, we want to push embeddings apart if their cosine similarity is above a margin.
        Args:
            Z (torch.Tensor): Node embeddings of shape (N, D).
            margin (float): Margin for repulsion.
            eps (float): Small value to avoid division by zero.
            If top_k=True: penalize only top-k most similar off-diagonal pairs per batch element.

        Returns:
            torch.Tensor: The computed cosine repulsion loss.
        """
        if Z.dim() == 2:
            Z = Z.unsqueeze(0)  # [1, C, D]

        B, C, D = Z.shape

        Z = Z / (Z.norm(dim=-1, keepdim=True) + eps)   # [B, C, D]
        G = Z @ Z.transpose(-1, -2)                    # [B, C, C]

        eye = torch.eye(C, device=Z.device, dtype=torch.bool).unsqueeze(0)  # [1, C, C]

        if top_k:
            if k is None:
                k = min(4 * C, C * (C - 1))

            G2 = G.masked_fill(eye, float("-inf"))       # remove diagonal
            vals = G2.reshape(B, -1)                     # [B, C*C]
            topk_vals, _ = torch.topk(vals, k=k, dim=-1) # [B, k]
            loss = (torch.relu(topk_vals - margin) ** 2).mean()
        else:
            G2 = G.masked_fill(eye, 0.0)                 # zero diagonal
            loss = (torch.relu(G2 - margin) ** 2).sum(dim=(-1, -2)).mean() / (C * (C - 1))

        return loss



    def get_div_weight(self,epoch, w_target, start=10, ramp=10):
        if epoch < start: return 0.0
        if epoch < start + ramp:
            return w_target * (epoch - start) / ramp
        return w_target

    

    def extra_loss_and_logs(self, pred, label, mask_value, loss_container, epoch):
        pred_dict = loss_container

        # loss = super(DSGNN_Engine, self).loss(pred, label, mask_value, loss_container)

        # if pred_dict['assignment_scores_source'] is None or pred_dict['assignment_scores_target'] is None:
        #     additional_loss = 0.0
        # else:
        #     mean_source_attention = torch.norm(pred_dict['assignment_scores_source'], p = 1)
        #     mean_target_attention = torch.norm(pred_dict['assignment_scores_target'], p = 1)

        #     additional_loss = self.additional_loss_weight * ( mean_source_attention + mean_target_attention)

        # if self.static_prefilter is not None:
        #     additional_loss /= self.model.num_contexts
        additional_loss = 0.0

        # DSN diversity regularizer to let DSN states be different
        dsn = pred_dict['dsn_states']
        div_loss = self.cosine_repulsion_loss(dsn['obs_augmented'], margin=self.dsn_div_margin, top_k=self.dsn_div_top_k)

        # total = loss + additional_loss + self.dsn_div_weight * div_loss
        div_loss_weight = self.get_div_weight(epoch, self.dsn_div_weight)
        total = additional_loss + div_loss_weight * div_loss

        # storring to log later
        add_logs = {
        'dsn_div_loss': div_loss.detach().item(),
        'dsn_div_loss_weighted': (self.dsn_div_weight * div_loss).detach().item(),
        'additional_loss': 0.0,#additional_loss.detach().item(),
            }


        return total, add_logs
    
    

    def log_val_obs_augmented_cosine_heatmap(self, epoch, log_every=5, max_contexts=300):
        """
        Logs a cosine-similarity heatmap of obs_augmented DSN states for ONE fixed val example.
        """
        if self._wandb_logger is None:
            return
        if not hasattr(self, "fixed_val_batch") or self.fixed_val_batch is None:
            return
        if (epoch % log_every) != 0:
            return

        # lazy import so training without wandb doesn't crash
        try:
            import wandb
        except Exception:
            return

        X = self.fixed_val_batch["X"].to(self._device)
        label = self.fixed_val_batch["label"].to(self._device)

        # Make sure val availability logic matches evaluate()
        # self.current_available_sensors = self._dataloader["val_loader"].available_sensors
        self.current_x_mask = self._to_device(self._to_tensor(self.fixed_val_batch["x_mask"]))
        self.current_label_mask = self._to_device(self._to_tensor(self.fixed_val_batch["label_mask"]))

        self.model.eval()
        with torch.no_grad():
            pred, label_in, pred_dict = self.forward(X, label, isTrain=False)

            if pred_dict is None or "dsn_states" not in pred_dict:
                return

            Z = pred_dict["dsn_states"]["obs_augmented"]  # [B, C, D]
            Z = Z[0]  # [C, D]

            # for readability / speed, subset DSNs if C is large
            C = Z.shape[0]
            if max_contexts is not None and C > max_contexts:
                idx = torch.linspace(0, C - 1, steps=max_contexts).long().to(Z.device)
                Z = Z.index_select(0, idx)
                C = max_contexts

            # cosine similarity
            Z = Z / (Z.norm(dim=-1, keepdim=True) + 1e-8)  # [C, D]
            G = Z @ Z.T                                    # [C, C]
            G_np = G.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(G_np, aspect="auto")
        ax.set_title(f"Val obs_augmented cosine sim (epoch {epoch})")
        ax.set_xlabel("DSN index")
        ax.set_ylabel("DSN index")
        # set value range to [0, 1]
        im.set_clim(0, 1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Log directly to wandb with step parameter for slider visualization
        wandb.log({
            "viz/obs_aug_cosine_heatmap": wandb.Image(fig)
        }, step=epoch)

        plt.close(fig)



    def evaluate(self, mode) -> tuple:
        evaluation_results = self.evaluate_custom(mode)

        if self.epoch in self.embedding_evolution or mode != 'val':
            return evaluation_results
        
        self.embedding_evolution[self.epoch] = self.model.context_emb_layer.detach().cpu().numpy()


        return evaluation_results
        

    def evaluate_custom(self, mode) -> tuple:
        if mode == 'test':
            self.load_model(self._save_path)
        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            self.current_available_sensors = self._dataloader[mode + '_loader'].available_sensors
            for X, label, x_mask, label_mask in tqdm(self._dataloader[mode + '_loader'].get_iterator(),total = self._dataloader[mode + '_loader'].num_batch, desc=f'{"Validation" if mode == "val" else "Test"}'):
                # X (b, t, n, f), label (b, t, n, 1)
                X, label = self._to_device(self._to_tensor([X, label]))


                # TODO: tidy up
                # Cache one fixed validation example for consistent visualization across epochs
                if mode == 'val' and not hasattr(self, "fixed_val_batch"):
                    self.fixed_val_batch = None

                if mode == 'val' and self.fixed_val_batch is None:
                    # Store a single example (batch size = 1) on CPU
                    self.fixed_val_batch = {
                        "X": self._take_first(X),
                        "label": self._take_first(label),
                        "x_mask": self._take_first(x_mask),
                        "label_mask": self._take_first(label_mask),
                    }
                # TODO: END

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
            # self.log_val_obs_augmented_cosine_heatmap(epoch=self.epoch + 1, log_every=1, max_contexts=500)
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
                    res = compute_all_metrics(preds[:,i,training_available_sensors.squeeze() == 1], labels[:,i,training_available_sensors.squeeze() == 1], mask_value)
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
                    mask_value
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
                    res = compute_all_metrics(preds[:,i,training_available_sensors.squeeze() == 0], labels[:,i,training_available_sensors.squeeze() == 0], mask_value)
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
                    mask_value
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


    def train(self):
        train_result = super().train()

        for keys in list(self.embedding_evolution.keys()):
            if keys > self.best_epoch:
                del self.embedding_evolution[keys]

        embedding_evolution_array = []
        for epoch in range(self.best_epoch + 1):
            embedding_evolution_array.append(self.embedding_evolution[epoch])

        np.save(self._save_path + '/embedding_evolution.npy', np.array(embedding_evolution_array))

        return train_result  