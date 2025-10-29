import os
import sys
import logging
from datetime import datetime

# Making wandb optional
# where wandb isn't installed. When not available, we simply ignore and disable wandb functionality 
try:
    import wandb  
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False


def get_logger(log_dir: str, name: str, log_filename: str, level: int = logging.INFO) -> logging.Logger:
    """Create or return a configured logger.

    - Ensures the log directory exists.
    - Avoids adding duplicate handlers when called multiple times for the same logger name.
    - Disables propagation to avoid double-logging.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(os.path.join(log_dir, log_filename), mode='a')
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.propagate = False

    # lightweight feedback for interactive runs
    print('Log directory:', log_dir)
    return logger


def get_run_name(args) -> str:
    """Generate a descriptive unique run name based on key args.
    
    Args:
        args: Argument parser namespace with relevant attributes.
    """
    run_name = f'{args.model_name}_{args.dataset}_{args.years}_{str(datetime.now().strftime("%Y-%m-%d %H:%M"))}'
    return run_name


class WandbLogger:
    """A small wrapper around wandb that makes wandb optional and easy to use.
    """

    def __init__(self, project: str, is_used: bool, name: str = None, entity: str = None, tags: list = None):
        """Initialize the wrapper. If `is_used` is True but wandb is not installed,
        the wrapper will disable itself and show a warning.

        Args:
            project: the wandb project name
            is_used: whether to use wandb (if False, this disables all wandb functionality)
            name: the wandb run name (optional)
            entity: the wandb entity (user or team) (optional)
        """
        self._requested = bool(is_used)
        if self._requested and not _WANDB_AVAILABLE:
            logging.getLogger(__name__).warning('wandb requested but not installed; disabling wandb logging')
        self.is_used = self._requested and _WANDB_AVAILABLE
        self._initialized = False
        self._project = project
        self._name = name
        self._entity = entity
        self._tags = tags

        if self.is_used:
            try:
                wandb.init(project=project, entity=entity, name=name, tags=tags)
                self._initialized = True
            except Exception as e:  # be defensive: don't let wandb failures kill the run
                logging.getLogger(__name__).exception('Failed to initialize wandb: %s', e)
                self.is_used = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finish()

    def finish(self):
        """Finish the wandb run (if initialized). Safe to call multiple times."""
        if self.is_used and self._initialized:
            try:
                wandb.finish()
            except Exception:
                logging.getLogger(__name__).exception('Error while finishing wandb run')
            finally:
                self.is_used = False
                self._initialized = False

    def watch_model(self, model, log: str = 'all', log_graph: bool = False):
        """Watch the given model using wandb. Parameters forwarded to `wandb.watch`.

        Args:
            model: the model (e.g., a torch.nn.Module)
            log: what to log (e.g., 'gradients', 'parameters', or 'all')
            log_graph: whether to attempt to log the model graph
        """
        if not self.is_used:
            return
        try:
            wandb.watch(model, log=log, log_graph=log_graph)
        except Exception:
            logging.getLogger(__name__).exception('wandb.watch failed')

    def log_hyperparams(self, params: dict):
        """Log hyper-parameters to wandb.config.
        
        Args:
            params: dictionary of hyper-parameters to log
        """
        if not self.is_used:
            return
        try:
            # allow changing values if needed
            wandb.config.update(params, allow_val_change=True)
        except Exception:
            logging.getLogger(__name__).exception('Failed to log hyperparameters to wandb')

    def log_metrics(self, metrics: dict):
        """Log a dict of metrics (timestamped by wandb).
        
        Args:
            metrics: dictionary of metric names to values
        """
        if not self.is_used:
            return
        try:
            wandb.log(metrics)
        except Exception:
            logging.getLogger(__name__).exception('Failed to log metrics to wandb')

    def log(self, key: str, value, round_idx: int = None):
        """Log a single scalar (optionally with a round/index field).
        
        Args:
            key: the metric name
            value: the metric value
            round_idx: optional round or index to log alongside the metric
            """
        if not self.is_used:
            return
        payload = {key: value}
        if round_idx is not None:
            payload['Round'] = round_idx
        try:
            wandb.log(payload)
        except Exception:
            logging.getLogger(__name__).exception('Failed to log %s to wandb', key)

    def log_str(self, key: str, value: str):
        """Log a string value to wandb (for notes or tags).
        
         Args:
            key: the name of the string field
            value: the string value to log
        """
        if not self.is_used:
            return
        try:
            wandb.log({key: value})
        except Exception:
            logging.getLogger(__name__).exception('Failed to log string %s to wandb', key)

    def save_file(self, path: str):
        """Save a local file to the wandb run. No-op if file doesn't exist.

        Args:
            path: the path to the file to save
        """
        if not self.is_used:
            return
        if path is None:
            return
        if not os.path.exists(path):
            logging.getLogger(__name__).warning('save_file called with non-existent path: %s', path)
            return
        try:
            # wandb.save accepts a path or pattern
            wandb.save(path)
        except Exception:
            logging.getLogger(__name__).exception('Failed to save file %s to wandb', path)