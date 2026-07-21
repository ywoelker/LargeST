import os
import argparse
import numpy as np
import uuid
from datetime import datetime


import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch

from src.models.sparsestategnn import SparseStateGNN
from src.engines.sparsestategnn_engine import SparseStateGNN_Engine
from src.utils.args import get_public_config, str2bool
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
    parser.add_argument('--n_hid', type=int, default=16)
    parser.add_argument('--n_context', type=int, default=32)
    parser.add_argument('--n_context_emb', type=int, default=32)
    parser.add_argument('--n_rand_dim', type=int, default=64)

    parser.add_argument('--static_prefilter_mode', type=str, default='static_dsn', choices=['none', 'static_dsn', 'identity', 'fixed'])
    parser.add_argument('--additional_loss_weight', type=float, default=0.0)

    parser.add_argument('--lrate', type=float, default=0.005)
    parser.add_argument('--wdecay', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--clip_grad_value', type=float, default=5.0)

    parser.add_argument('--model_description', type=str, default='SparseStateGNN')

    parser.add_argument('--dsn_div_weight', type=float, default=0.1)
    parser.add_argument('--dsn_div_margin', type=float, default=0.2)
    parser.add_argument('--dsn_div_top_k', type=str2bool, default=True)

    parser.add_argument('--gcn_layers', type=int, default=3)
    parser.add_argument('--adding_query_to_dsn', type=str2bool, default=True)

    parser.add_argument('--attention_method', type=str, default='MLA', choices=['MLA', 'MHA'])

    parser.add_argument('--pos_enc_dim', type=int, default=16, help='Dimension of sinusoidal positional encoding for timestep index.')

    args = parser.parse_args()
    args.model_name = 'SparseStateGNN' if args.model_name == '' else args.model_name

    log_dir = './results/{}/{}/{}_{}/'.format(args.dataset, args.model_name, datetime.now().strftime('%m-%d_%H-%M-%S'), str(uuid.uuid4())[-6:])
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)

    return args, log_dir, logger


def create_prefilter_with_fixed_context_counter(n_sensors, n_hidden_states):
    prefilter_matrix = np.zeros((n_sensors, n_hidden_states))
    filled_columns = 0
    while filled_columns < n_sensors:
        matrix = np.eye(n_hidden_states)
        delta = min(n_sensors - filled_columns, n_hidden_states)
        prefilter_matrix[filled_columns: filled_columns + delta] = matrix[:delta]
        filled_columns += n_hidden_states
    return prefilter_matrix


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)

    from src.utils.logging import WandbLogger, get_run_name
    wandb_logger = WandbLogger(project=args.wandb_project,
                               is_used=args.use_wandb,
                               name=get_run_name(args),
                               entity=args.wandb_entity,
                               tags=args.wandb_tags,
                               )
    wandb_logger.log_hyperparams(vars(args))

    data_path, _, node_num = get_dataset_info(args.dataset)

    dataloader, scaler = load_dataset(data_path, args, logger, drop_unavailable_sensors=True)

    if args.static_prefilter_mode == 'static_dsn':
        static_assignment = np.load(os.path.join(data_path, args.years, 'static_assignment.npz'))['static_assignment_matrices']
        static_dsn_count = static_assignment.shape[1]
        args.n_context = static_dsn_count

    elif args.static_prefilter_mode == 'identity':
        static_assignment = np.eye(node_num)
        args.n_context = node_num

    elif args.static_prefilter_mode == 'fixed':
        static_assignment = create_prefilter_with_fixed_context_counter(node_num, args.n_context)

    else:
        static_assignment = None

    model = SparseStateGNN(
        **{
            "num_nodes": node_num,
            "seq_num": args.seq_len,
            "in_dim": args.input_dim,
            "out_dim": args.horizon,
            "random_feature_dim": args.n_rand_dim,
            "node_emb_dim": args.n_context_emb,
            "layer_num": args.gcn_layers,
            "adding_query_to_dsn": args.adding_query_to_dsn,
            "time_emb_dim": 8,
            "use_residual": True,
            "use_bn": True,
            "use_spatial": False,
            "hid_dim": args.n_hid,
            "n_contexts": args.n_context,
            "dropout": args.dropout,
            "attention_method": args.attention_method,
            "time_of_day_size": 288,
            "day_of_week_size": 7,
            "pos_enc_dim": args.pos_enc_dim,
        }
    )

    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 25, 50], gamma=0.5)

    if type(static_assignment) is np.ndarray:
        static_assignment = torch.tensor(static_assignment, dtype=torch.float32, device=device)

    engine = SparseStateGNN_Engine(device=device,
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
                        static_prefilter=static_assignment,
                        patience=args.patience,
                        log_dir=log_dir,
                        logger=logger,
                        seed=args.seed,
                        wandb_logger=wandb_logger,
                        training_timeout_min=args.training_timeout_min,
                        additional_loss_weight=args.additional_loss_weight,
                        dsn_div_weight=args.dsn_div_weight,
                        dsn_div_margin=args.dsn_div_margin,
                        dsn_div_top_k=args.dsn_div_top_k,
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
