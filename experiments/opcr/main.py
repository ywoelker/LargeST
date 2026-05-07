import os
import argparse
import numpy as np
import uuid
from datetime import datetime


import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.opcr.opcr import Model as OPCR
from src.engines.opcr_engine import OPCR_Engine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    parser.add_argument('--adj_threshold', type=float, default=0.1)

    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=0)
    parser.add_argument('--clip_grad_value', type=float, default=5)
    args = parser.parse_args()

    args.model_name = 'OPCR' if args.model_name == '' else args.model_name

    log_dir = './results/{}/{}/{}_{}/'.format(args.dataset, args.model_name,datetime.now().strftime('%m-%d_%H-%M-%S'), str(uuid.uuid4())[-6:])
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    
    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)

    # Initialize wandb
    from src.utils.logging import WandbLogger, get_run_name
    wandb_logger = WandbLogger(project=args.wandb_project, 
                               is_used=args.use_wandb, 
                               name=get_run_name(args),
                               entity=args.wandb_entity,
                               tags=args.wandb_tags,
                               )
    wandb_logger.log_hyperparams(vars(args))

    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    logger.info('Adj path: ' + adj_path)
    
    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx[adj_mx < args.adj_threshold] = 0
    adj_mx = np.maximum.reduce([adj_mx, adj_mx.T]) # force to be symmetric

    idxs = np.nonzero(adj_mx)
    edge_index = np.stack(idxs)
    edge_index = torch.tensor(edge_index).to(device)
    
    dataloader, scaler = load_dataset(data_path, args, logger)


    model = OPCR(
        num_nodes = node_num,
        in_len = args.seq_len,
        in_dim= args.input_dim - 2,
        hidden_dim = 128,
        s_layers = 2,
        num_layers= 2,
        time_dim = 2, 
        device = device,
        edge_index= edge_index,
        dropout=0,
        out_dim= 1,
        horizon= args.horizon
    )

    loss_fn = masked_mae
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = None# torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.1, verbose=True)


    engine = OPCR_Engine(device=device,
                          model=model,
                          dataloader=dataloader,
                          scaler=scaler,
                          sampler=None,
                          loss_fn=loss_fn,
                          lrate=args.lrate,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          clip_grad_value=args.clip_grad_value,
                          max_epochs=args.max_epochs,
                          patience=args.patience,
                          log_dir=log_dir,
                          logger=logger,
                          seed=args.seed,
                          wandb_logger=wandb_logger,
                          training_timeout_min=args.training_timeout_min
                          )

    if args.mode == 'train':
        try:
            engine.train()
        except KeyboardInterrupt:
            logger.info('Exiting from training early')
            logger.info('Evaluating using the best model found so far')
            engine.evaluate('test')
    else:
        engine.evaluate(args.mode)


if __name__ == "__main__":
    main()