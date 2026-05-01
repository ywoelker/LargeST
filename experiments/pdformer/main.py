import os
import argparse
import numpy as np
import uuid
from datetime import datetime

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch

from src.models.pdformer.pdformer import PDFormer
from src.engines.pdformer_engine import PDFormer_Engine
from src.utils.args import get_public_config, str2bool
from src.models.pdformer.dataloader import load_dataset, get_dataset_info
from src.utils.logging import get_logger


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()

    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--skip_dim', type=int, default=256)
    parser.add_argument('--lape_dim', type=int, default=8)
    parser.add_argument('--geo_num_heads', type=int, default=4)
    parser.add_argument('--sem_num_heads', type=int, default=2)
    parser.add_argument('--t_num_heads', type=int, default=2)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--qkv_bias', type=str2bool, default=True)
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--attn_drop', type=float, default=0.0)
    parser.add_argument('--drop_path', type=float, default=0.3)
    parser.add_argument('--s_attn_size', type=int, default=3)
    parser.add_argument('--t_attn_size', type=int, default=3)
    parser.add_argument('--enc_depth', type=int, default=6)
    parser.add_argument('--type_ln', type=str, default='pre')
    parser.add_argument('--type_short_path', type=str, default='dist', choices=['dist', 'hop'])

    parser.add_argument('--lrate', type=float, default=0.001)
    parser.add_argument('--wdecay', type=float, default=0.0001)
    parser.add_argument('--clip_grad_value', type=float, default=5.0)

    parser.add_argument('--use_curriculum_learning', type=str2bool, default=True)
    parser.add_argument('--step_size', type=int, default=2500)
    parser.add_argument('--max_epoch', type=int, default=200)

    parser.add_argument('--far_mask_delta', type=float, default=0.25)
    parser.add_argument('--dtw_delta', type=int, default=5)
    parser.add_argument('--random_flip', type=str2bool, default=True)
    parser.add_argument('--set_loss', type=str, default='masked_mae')

    parser.add_argument('--cand_key_days', type=int, default=21)
    parser.add_argument('--n_cluster', type=int, default=16)
    parser.add_argument('--cluster_max_iter', type=int, default=5)
    parser.add_argument('--cluster_method', type=str, default='kshape', choices=['kshape', 'softdtw'])
    parser.add_argument('--time_intervals', type=int, default=300)
    parser.add_argument('--weight_adj_epsilon', type=float, default=0.1)

    parser.add_argument('--add_time_in_day', type=str2bool, default=False)
    parser.add_argument('--add_day_in_week', type=str2bool, default=False)

    parser.add_argument('--model_description', type=str, default='pdformer')

    args = parser.parse_args()
    args.model_name = 'PDFormer' if args.model_name == '' else args.model_name

    log_dir = './results/{}/{}/{}_{}/'.format(
        args.dataset,
        args.model_name,
        datetime.now().strftime('%m-%d_%H-%M-%S'),
        str(uuid.uuid4())[-6:]
    )
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)

    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)

    from src.utils.logging import WandbLogger, get_run_name
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        is_used=args.use_wandb,
        name=get_run_name(args),
        entity=args.wandb_entity,
        tags=args.wandb_tags,
    )
    wandb_logger.log_hyperparams(vars(args))

    data_path, _, node_num = get_dataset_info(args.dataset)

    dataloader, scaler, data_feature = load_dataset(
        data_path, args, logger, drop_unavailable_sensors=False
    )

    model_config = vars(args).copy()
    model_config['device'] = device
    model_config['num_nodes'] = node_num
    model_config['input_window'] = args.seq_len
    model_config['output_window'] = args.horizon
    model_config['max_epoch'] = args.max_epochs

    model = PDFormer(model_config, data_feature)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lrate,
        weight_decay=args.wdecay,
        eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[25, 50],
        gamma=0.5
    )

    engine = PDFormer_Engine(
        device=device,
        model=model,
        dataloader=dataloader,
        scaler=scaler,
        sampler=None,
        loss_fn=None,
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
        lape_dim=args.lape_dim,
        random_flip=args.random_flip,
        set_loss=args.set_loss,
        training_timeout_min=args.training_timeout_min,
        
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