from src.utils.metrics import compute_all_metrics
import torch

class BaseExperiment():

    def __init__(self, name, description=None):
        self.name = name
        self.description = description
        

    def train_preprocess(self, X, label, label_mask_value = -torch.inf):
        """
        Preprocess the input data and labels.
        This method can be overridden by subclasses for custom preprocessing.
        """
        return X, label
    

    def eval_preprocess(self, X, label, label_mask_value = -torch.inf):
        """
        Preprocess the input data and labels for evaluation.
        This method can be overridden by subclasses for custom preprocessing.
        """
        return X, label
    
    def experiment_evaluation_metrics (self, preds, labels, mask_value):
        """
        Compute evaluation metrics.
        This method can be overridden by subclasses for custom metrics.

        # Returns:
        - A dictionary of each way of computing the experiment dependend metrics. Each value should contain a list with three elements MAE, MAPE, RMSE
        """
        return {}
    


class SparsityExperiment(BaseExperiment):
    """
    An experiment class that handles sparsity in the data.
    It can be used to preprocess data and labels for training and evaluation.
    """
    
    def __init__(self, name: str, description: str, 
                 input_sparseness: str, input_dropout: float, 
                 output_sparseness: str, output_dropout: float, 
                 train_dropout: float, seed: int, n_sensors: int, device):
        super().__init__(name, description)

        self.input_sparseness = input_sparseness
        self.input_dropout = input_dropout
        self.output_sparseness = output_sparseness
        self.output_dropout = output_dropout
        self.train_dropout = train_dropout
        self.seed = seed
        self.n_sensors = n_sensors

        self.mask_iter = 0


        if self.train_dropout > 0:
            # Dropout some sensors which will only appear in the training process

            mask_shape = (n_sensors, )
            mask_tensor = torch.rand(mask_shape, dtype=torch.float32, device=device) > self.train_dropout
            self.train_mask = mask_tensor.reshape(1, 1, n_sensors, 1)
        else:
            self.train_mask = torch.ones((1, 1, n_sensors, 1), dtype=torch.float32, device=device)   

    def _get_mask(self, sparseness_type: str, dropout: str, data):
        """
        Generate a mask based on the specified sparsity type and dropout rate.
        """
        b, t, n, f = data.shape
        if sparseness_type == 'point':
            mask_shape = (b, t, n)
        elif sparseness_type == 'spatial':
            mask_shape = (b, n)
        else:
            raise ValueError("Invalid sparsity type. Use 'point' or 'spatial'.")
        
        mask_tensor = torch.rand(mask_shape, dtype=torch.float32, device=data.device) > dropout
        mask_tensor = mask_tensor.unsqueeze(-1)

        if sparseness_type == 'spatial':
            mask_tensor = mask_tensor.unsqueeze(1)

        return mask_tensor.float()
    
    def _set_seed_for_mask(self):
        torch.manual_seed(self.seed + self.mask_iter)
        self.mask_iter += 1
        
    
    def train_preprocess(self, X, label, label_mask_value = -torch.inf):
        b, t, n, f = X.shape
        if self.input_sparseness != 'none' and self.input_dropout > 0:
            # Randomly mask input features

            mask_tensor = self._get_mask(self.input_sparseness, self.input_dropout, X)           
            X = X * mask_tensor # Apply mask to the features
        
        if self.output_sparseness != 'none' and self.output_dropout > 0:
            # Randomly mask output features
            mask_tensor = self._get_mask(self.output_sparseness, self.output_dropout, label)
            label = torch.where(mask_tensor == 0, label_mask_value, label)
        
    
        if self.train_dropout > 0:
            X = X * self.train_mask  # Apply train mask to the features
            label = torch.where(self.train_mask.expand(b,t,n,f) == 0, label_mask_value, label)


        return X, label
    
    def eval_preprocess(self, X, label, label_mask_value = -torch.inf):

        self._set_seed_for_mask()
        b, t, n, f = X.shape
        if self.input_sparseness != 'none' and self.input_dropout > 0:
            # Randomly mask input features

            mask_tensor = self._get_mask(self.input_sparseness, self.input_dropout, X)           
            X = X * mask_tensor  # Apply mask to the features


        if self.output_sparseness != 'none' and self.output_dropout > 0:
            # Randomly mask output features
            mask_tensor = self._get_mask(self.output_sparseness, self.output_dropout, label)
            label = torch.where(mask_tensor == 0, label_mask_value, label)
    
        return X, label
    

    
    def experiment_evaluation_metrics(self, preds, labels, mask_value):
        
        additional_metrics = {}

        if self.train_dropout > 0:
            training_bool_mask = self.train_mask.squeeze().to(torch.bool).to(preds.device)
            preds = preds[ :, ~training_bool_mask]
            labels = labels[ :, ~training_bool_mask]

            # Compute metrics
            metric = compute_all_metrics(preds, labels, mask_value)
            additional_metrics['training_mask'] = metric

        return additional_metrics


