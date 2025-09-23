import argparse

def get_public_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    # if need to use the data from multiple years, please use underline to separate them, e.g., 2018_2019
    parser.add_argument('--years', type=str, default='2019')
    parser.add_argument('--model_name', type=str, default='')
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
    return parser