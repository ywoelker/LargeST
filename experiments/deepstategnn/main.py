import os
import argparse
import numpy as np
import uuid
from datetime import datetime


import sys
sys.path.append(os.path.abspath(__file__ + '/../../..'))

import torch
torch.set_num_threads(3)

from src.models.deepstategnn import TrafficReconGNN
from src.engines.deepstategnn_engine import DeepStateGNN_Engine 
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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def get_config():
    parser = get_public_config()
    parser.add_argument("--num_temporal_metanodes",
                        type=int,
                        default=5,
                        help="The number of temporal metanodes.")
    parser.add_argument("--num_airquality_metanodes",
                        type=int,
                        default=5,
                        help="The number of airquality metanodes.")
    parser.add_argument("--num_weather_metanodes",
                        type=int,
                        default=5,
                        help="The number of weather metanodes.")
    parser.add_argument("--nodes_hidden_dim",
                        type=int,
                        default=32,
                        help="The hidden dimension of the nodes.")
    parser.add_argument("--edge_hidden_dim",
                        type=int,
                        default=16,
                        help="The hidden dimension of the edges.")
    parser.add_argument("--graph_embedding_dim",
                        type=int,
                        default=32,
                        help="The dimension of the graph embedding.")
    parser.add_argument("--observation_hidden_dim",
                        type=int,
                        default=16,
                        help="The hidden dimension of the observation while the embedding process.")
    parser.add_argument("--attn_heads",
                        type=int,
                        default=8,
                        help="The number of attention heads.")
    parser.add_argument("--assignment_threshold",
                        type=float,
                        default=0.015,
                        help="The threshold score to not assign observations to meta-nodes.")
    parser.add_argument("--spatial_assignment_threshold",
                        type=float,
                        default=0.8,
                        help="The threshold score to not assign observations to meta-nodes with spatial attributes.")
    parser.add_argument("--num_gru_layers",
                        type=int,
                        default=2,
                        help="The number of GRU layers.")
    parser.add_argument("--observation_loss_weight",
                        type=float,
                        default=1.0,
                        help="The weight of the observation loss.")
    parser.add_argument("--densestatenode_loss",
                        type=float,
                        default=1.0,
                        help="The weight of the dense state node loss.")
    parser.add_argument("--alpha",
                        type=float,
                        default=0.5,
                        help="The alpha parameter for the laplacian weighting.")
    parser.add_argument("--use_temporal_dsn",
                        type=str2bool,
                        default=True,
                        help="Whether to use temporal dense state nodes.")
    parser.add_argument("--use_airquality_dsn",
                        type=str2bool,
                        default=False,
                        help="Whether to use airquality dense state nodes.")
    parser.add_argument("--use_weather_dsn",
                        type=str2bool,
                        default=False,
                        help="Whether to use weather dense state nodes.")
    parser.add_argument("--use_semantic_dsn",
                        type=str2bool,
                        default=True,
                        help="Whether to use semantic dense state nodes.")
    parser.add_argument("--use_spatial_dsn",
                        type=str2bool,
                        default=True,
                        help="Whether to use spatial dense state nodes.")
    parser.add_argument("--use_streest_as_dsn",
                        type=str2bool,
                        default=True,
                        help="Whether to use streets as dense state nodes.")
    parser.add_argument("--use_gcn_layer",
                        type=str2bool,
                        default=True,
                        help="Whether to use GCN layer.")
    parser.add_argument("--use_message_passing",
                        type=str2bool,
                        default=True,
                        help="Whether to use message passing.")
    parser.add_argument('--aggregation_method', 
                        type=str,
                        default='attbigru',
                        choices=['mean', 'sum', 'max', 'wmean', 'meanstd', 'transformer', 'attbigru', 'multihead', 'meanminmax'],                
                        help='The aggregation method for the graph nodes.')
    parser.add_argument('--global_embedding',
                        type=str2bool,
                        default=True,
                        help='Whether to use global embedding or if each DSN type has its own embedding.')
    parser.add_argument('--num_dynamical_dsns',
                        type=int,
                        default=0,
                        help='The number of dynamical dense state nodes.')
    parser.add_argument('--context_in_embedding',
                        type=str2bool,
                        default=True,
                        help='Whether to include the context in the embedding process.')
    parser.add_argument('--zero_temporal_dsn_after_aggregation',
                        type=str2bool,
                        default=False,
                        help='Whether to zero out the temporal dense state nodes after the aggregation.')
    parser.add_argument('--zero_env_dsn_after_aggregation',
                        type=str2bool,
                        default=False,
                        help='Whether to zero out the airquality dense state nodes after the aggregation.')
    parser.add_argument('--zero_spatial_dsn_after_aggregation',
                        type=str2bool,
                        default=False,
                        help='Whether to zero out the spatial dense state nodes after the aggregation.')
    parser.add_argument('--zero_semantic_dsn_after_aggregation',
                        type=str2bool,
                        default=False,
                        help='Whether to zero out the semantic dense state nodes after the aggregation.')
    
    parser.add_argument('--detach_query_embedding',
                        type=str2bool,
                        default=False,
                        help='Whether to detach the query embedding.')
    parser.add_argument('--detach_loss_embedding',
                        type=str2bool,
                        default=False,
                        help='Whether to detach the query embedding.')

    parser.add_argument('--lrate', type=float, default=0.002)
    parser.add_argument('--wdecay', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--clip_grad_value', type=float, default=5.0)
    
    args = parser.parse_args()

    args.model_name = 'DeepStateGNN' if args.model_name == '' else args.model_name

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
                               entity=args.wandb_entity
                               )
    wandb_logger.log_hyperparams(vars(args))
    
    data_path, adj_path, node_num = get_dataset_info(args.dataset)
    logger.info('Adj path: ' + adj_path)

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = normalize_adj_mx(adj_mx, 'doubletransition')
    args.adjs = [torch.tensor(i).to(device) for i in adj_mx]
    
    dataloader, scaler = load_dataset(data_path, args, logger)




    train_locations = dataloader['train_loader'].metadata[:, :2]

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=14, random_state=args.seed)
    kmeans.fit(train_locations)


    neighbourhood_shapes = {
        'features' : [],
    }


    for center in kmeans.cluster_centers_:
        neighbourhood_shapes['features'].append({
            'geometry': {
                'type': 'Polygon',
                'coordinates': np.array([[center]])
            }
        })

    print('Neighbourhood shapes:', neighbourhood_shapes)




    model = TrafficReconGNN(
        nodes_hidden_dim=args.nodes_hidden_dim, 
        edge_embedding_dim=args.edge_hidden_dim, 
        graph_embedding_dim=args.graph_embedding_dim, 
        observation_hidden_dim= args.observation_hidden_dim,
        query_horizon=args.horizon, 
        num_attention_heads=args.attn_heads, 
        neighborhood_shapes=neighbourhood_shapes, 
        num_temporal_metanodes=args.num_temporal_metanodes,  
        num_aq_metanodes=args.num_airquality_metanodes, 
        num_weather_metanodes=args.num_weather_metanodes, 
        metadata_dict=dataloader['train_loader'].metadata_dict[()],
        num_gru_layers=args.num_gru_layers,
        assignment_threshold=args.assignment_threshold ,
        alpha=args.alpha,  
        detach_query_embedding = args.detach_query_embedding,
        detach_loss_embedding = args.detach_loss_embedding,
        use_temporal_dsn=args.use_temporal_dsn,
        use_aq_dsn=args.use_airquality_dsn,
        use_weather_dsn=args.use_weather_dsn,
        use_semantic_dsn=args.use_semantic_dsn,
        use_spatial_dsn=args.use_spatial_dsn,
        use_streest_as_dsn = args.use_streest_as_dsn,
        use_gcn_layer=args.use_gcn_layer,
        aggregation_methods = args.aggregation_method,
        dropout = args.dropout,
        global_embedding= args.global_embedding,
        num_dynamical_dsns = args.num_dynamical_dsns,
        spatial_assignment_threshold= args.spatial_assignment_threshold,
        use_message_passing = args.use_message_passing,
        context_in_embedding = args.context_in_embedding,
        )

    loss_fn = masked_mae
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.wdecay, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 50], gamma=0.5)

    engine = DeepStateGNN_Engine(device=device,
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
                        wandb_logger=wandb_logger
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