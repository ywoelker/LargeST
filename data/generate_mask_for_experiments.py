import numpy as np
import pandas as pd
from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path
import torch

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    arg_parser = ArgumentParser()


    arg_parser.add_argument('--dataset', type = str, required=True, help='Name of the dataset (e.g., "ca", "gba", "gla", "sd")')

    arg_parser.add_argument('--input_dropout', type = float, default=0.0, help='Input dropout rate (between 0 and 1)')
    arg_parser.add_argument('--input_fault', type=float, default=0.0, help='Input fault rate (between 0 and 1)')
    arg_parser.add_argument('--input_fault_minmax', type=int, nargs=2, default=[12, 48], help='Min and max length for input fault injection')
    arg_parser.add_argument('--input_sensor_dropout', type=float, default=0.0, help='Input sensor dropout rate (between 0 and 1)')

    arg_parser.add_argument('--output_dropout', type = float, default=0.0, help='Output dropout rate (between 0 and 1)')
    arg_parser.add_argument('--output_fault', type=float, default=0.0, help='Output fault rate (between 0 and 1)')
    arg_parser.add_argument('--output_fault_minmax', type=int, nargs=2, default=[12, 48], help='Min and max length for output fault injection')
    arg_parser.add_argument('--output_sensor_dropout', type=float, default=0.0, help='Output sensor dropout rate (between 0 and 1)')

    arg_parser.add_argument('--share_input_output_mask', type = str2bool, default=False, help='Whether to have one mask for the complete dataset and take the X and y from the same dropouted data')

    arg_parser.add_argument('--train_dropout', type = float, default=0.0, help='Training sensor dropout rate (between 0 and 1)')

    arg_parser.add_argument('--seed', type = int, default=42, help='Random seed for reproducibility')

    arg_parser.add_argument('--experiment_folder', type = str, required=True, help='Name of the experiment for the folder')
    arg_parser.add_argument('--n_masks', type=int, default=1, help='Number of different masks to generate')


    return arg_parser.parse_args()

def _get_mask(
        n_samples: int, n_sensors: int, 
        dropout: float, 
        fault_rate:float = 0.0, fault_minmax = (12, 48), 
        sensor_dropout: float = 0.0
        ):
    """
    Generate a mask based on the specified sparsity type and dropout rate.
    1 for each entry that is observed, and 0 if it is missing.
    """

    shape = (n_samples, n_sensors)


    mask = torch.ones(shape, dtype=torch.bool)


    if dropout > 0:
        random_drop = torch.rand(shape) > dropout
        mask = mask & random_drop


    if fault_rate > 0:
        fault_len_min  = fault_minmax[0]
        fault_len_max  = fault_minmax[1]

        fault_mask = torch.rand(shape) < fault_rate

        for sensor_i in range(n_sensors):
            fault_starts = torch.where(fault_mask[:, sensor_i])[0].tolist()
            
            for fault_start in fault_starts:
                fault_len = np.random.randint(fault_len_min, fault_len_max+1)
                fault_end = min(fault_start + fault_len, n_samples)
                fault_mask[fault_start:fault_end, sensor_i] = True

        fault_mask = ~fault_mask # Turn it around that 1 is observed and 0 is missing

        mask = mask & fault_mask


    if sensor_dropout > 0:
        sensor_drop = (torch.rand((n_sensors,)) > sensor_dropout).to(torch.bool)
        mask = mask & sensor_drop.unsqueeze(0).expand_as(mask)

    return mask


def main():
    args = parse_arguments()
    np.random.seed(args.seed)


    ### find the set of sensors for the dataset
    data_path = Path('data/') / args.dataset
    ptr = np.load(data_path / '2019' / 'his.npz', allow_pickle=True)
    n_samples, n_sensors, n_features = ptr['data'].shape

    experiment_path = Path('data/masks') / args.dataset / args.experiment_folder

    for mask_i in range(args.n_masks):

        mask_folder = experiment_path / f'mask_{mask_i:02d}'
        mask_folder.mkdir(parents=True, exist_ok=True)

        input_mask = _get_mask(
            n_samples, n_sensors, 
            dropout=args.input_dropout, 
            fault_rate=args.input_fault, 
            fault_minmax = args.input_fault_minmax,
            sensor_dropout= args.input_sensor_dropout
            )
        torch.save(input_mask, mask_folder / 'input_mask.pt')
        
        if args.share_input_output_mask:
            output_mask = input_mask.clone()

        else:
            output_mask = _get_mask(
                n_samples, n_sensors, 
                dropout=args.output_dropout, 
                fault_rate=args.output_fault, 
                fault_minmax = args.output_fault_minmax,
                sensor_dropout=args.output_sensor_dropout
            )
        torch.save(output_mask, mask_folder / 'output_mask.pt')


        if args.train_dropout > 0:
            train_mask = (torch.rand((1, n_sensors)) > args.train_dropout).to(torch.bool)
            torch.save(train_mask, mask_folder / 'train_mask.pt')

        print(f'Missing rate input: {1 - input_mask.float().mean().item():.4f}, output: {1 - output_mask.float().mean().item():.4f}, train: {1 - train_mask.float().mean().item() if args.train_dropout > 0 else 0:.4f}')


    # save args in a config json
    import json
    with open(experiment_path / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)


if __name__ == "__main__":
    main()

