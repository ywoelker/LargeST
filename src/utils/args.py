import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_public_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    # if need to use the data from multiple years, please use underline to separate them, e.g., 2018_2019
    parser.add_argument('--years', type=str, default='2019')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--run_description', type=str, default='')
    parser.add_argument('--seed', type=int, default=2023)

    parser.add_argument('--bs', type=int, default=64)
    # seq_len denotes input history length, horizon denotes output future length
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--output_dim', type=int, default=1)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=30)


    parser.add_argument('--mask_name', type=str, default=None, help='Name to the folder where the mask member is stored. If None, no mask is used.')
    parser.add_argument('--mask_iter', type= int, default=0, help='Which mask to use from the mask folder.')
    parser.add_argument('--use_metadata', type=bool, default=False, help='Whether to use the metadata (e.g., time of day, day of week, etc.)')

    parser.add_argument('--train_data_percentage', type=float, default=1.0, help='Percentage of training data to use (between 0 and 1)')
    parser.add_argument('--training_timeout_min', type=int, default=None, help='Timeout for the training phase in minutes. If None or 0, no timeout is implied.')


    

    # Add wandb arguments
    parser.add_argument('--use_wandb', type=str2bool, default=True,
                        help='Whether to use wandb logging')
    parser.add_argument('--wandb_project', type=str, default='deepstategnn',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default='usc-infolab',
                       help='Weights & Biases entity (username or team)')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=[],
                       help='Tags for the wandb run')
    return parser