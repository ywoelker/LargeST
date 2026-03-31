import os
import argparse
import numpy as np
import uuid
from datetime import datetime


import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.graphsparsenet.model import GSNet
from src.base.engine import BaseEngine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, load_adj_from_numpy, get_dataset_info
from src.utils.graph_algo import normalize_adj_mx
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
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--Ks', type=int, default=3)
    parser.add_argument('--block_num', type=int, default=2)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)

    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--clip_grad_value', type=float, default=0)
    args = parser.parse_args()

    args.model_name = 'GSNet' if args.model_name == '' else args.model_name

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

    dataloader, scaler = load_dataset(data_path, args, logger)

    model = GSNet(
        device=device,
        seq_num=args.seq_len,
        in_dim=args.input_dim,
        hid_dim=128,
        num_nodes=node_num,
        dropout=0.1,
        tau = 0.25,
        random_feature_dim=64,
        node_emb_dim=128,
        time_emb_dim=32,
        use_residual=True,
        use_bn=True,
        use_spatial=False,
        use_long=False,
        supports=None,
        edge_indices=None
    )
    
    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    engine = BaseEngine(device=device,
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