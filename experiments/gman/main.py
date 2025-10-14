import os
import argparse
import numpy as np
import uuid
from datetime import datetime

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.gman import GMAN
from src.engines.gman_engine import GMAN_Engine
from src.utils.args import get_public_config
from src.utils.dataloader import load_dataset, get_dataset_info
from src.utils.metrics import masked_mae
from src.utils.logging import get_logger, WandbLogger, get_run_name


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()
    # ---- GMAN hyperparams ----
    parser.add_argument('--L', type=int, default=3, help='# STAtt blocks per encoder/decoder')
    parser.add_argument('--K', type=int, default=4, help='# attention heads')
    parser.add_argument('--d', type=int, default=16, help='dim of each head (D=K*d)')
    parser.add_argument('--bn_decay', type=float, default=0.1)

    # ---- opt / sched ----
    parser.add_argument('--lrate', type=float, default=1e-3)
    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--clip_grad_value', type=float, default=5)

    args = parser.parse_args()
    args.model_name = 'GMAN' if args.model_name == '' else args.model_name

    # Keep LargeST style logging path
    log_dir = './results/{}/{}/{}_{}/'.format(
        args.dataset, args.model_name,
        datetime.now().strftime('%m-%d_%H-%M-%S'), str(uuid.uuid4())[-6:]
    )
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)

    # ---- wandb ----
    wandb_logger = WandbLogger(project=args.wandb_project,
                               is_used=args.use_wandb,
                               name=get_run_name(args),
                               entity=args.wandb_entity)
    wandb_logger.log_hyperparams(vars(args))

    # ---- data ----
    data_path, _, node_num = get_dataset_info(args.dataset)
    dataloader, scaler = load_dataset(data_path, args, logger)

    # ---- model ----
    # GMAN expects SE (learnable) + args + bn_decay.
    D = args.K * args.d
    SE = torch.nn.Parameter(torch.randn(node_num, D))  # learnable spatial embeddings
    model = GMAN(SE=SE, args=args, bn_decay=args.bn_decay)

    # add attributes used by BaseEngine / eval
    setattr(model, "param_num", lambda: sum(p.numel() for p in model.parameters()))

    # ---- opt/sched ----
    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)

    # ---- engine ----
    engine = GMAN_Engine(device=device,
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
                         wandb_logger=wandb_logger)

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