import os
import argparse
import numpy as np
import uuid
from datetime import datetime

import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.bigst_deepstate_dev import BigST
from src.engines.bigst_deepstate_dev_engine import BigST_Engine
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

"""
"num_nodes": 170,
    "in_dim": 3,
    "dropout": 0.3,
    "input_length": INPUT_LEN,
    "output_length": OUTPUT_LEN,
    "nhid": 32,
    "tiny_batch_size": 64,

"""
def get_config():
    parser = get_public_config()
    parser.add_argument('--nhid', type=int, default=32)
    parser.add_argument('--tiny_batch_size', type=int, default=64)

    parser.add_argument('--lrate', type=float, default=0.002)
    parser.add_argument('--wdecay', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--clip_grad_value', type=float, default=5.0)

    parser.add_argument('--model_description', type=str, default='bigst_deepstate_dev')
    
    args = parser.parse_args()

    log_dir = './results/{}/{}/{}_{}/'.format(args.dataset, args.model_name,datetime.now().strftime('%m-%d_%H-%M-%S'), str(uuid.uuid4())[-6:])
    logger = get_logger(log_dir, __name__, 'record_s{}.log'.format(args.seed))
    logger.info(args)
    
    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()
    set_seed(args.seed)
    device = torch.device(args.device)
    
    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    logger.info('Adj path: ' + adj_path)

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, 'doubletransition')
    args.adjs = [torch.tensor(i).to(device) for i in adj_mx]
    
    dataloader, scaler = load_dataset(data_path, args, logger)

    model = BigST(
        bigst_args={
                "num_nodes": node_num,
                "seq_num": args.seq_len, 
                "in_dim": args.input_dim,
                "out_dim": args.horizon,
                "hid_dim": 32,
                "tau" : 0.25,
                "random_feature_dim": 64,
                "node_emb_dim": 32,
                "time_emb_dim": 32,
                "use_residual": True,
                "use_bn": True,
                "use_long": False,
                "use_spatial": False,
                "dropout": 0.3,
                "supports": args.adjs,
                "time_of_day_size": 96, 
                "day_of_week_size": 7},preprocess_path= None, preprocess_args=None
                    )
    
    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 50], gamma=0.5)

    
    engine = BigST_Engine(device=device,
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
                        seed=args.seed
                        )

    if args.mode == 'train':
        engine.train()
    else:
        engine.evaluate(args.mode)


if __name__ == "__main__":
    main()