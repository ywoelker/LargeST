import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from typing import Optional
import pickle as pckl 

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def generate_metadata(metadata, add_location, add_road, add_region, add_lanes, add_direction):

    min_lat, max_lat = metadata['Lat'].min(), metadata['Lat'].max()
    min_lng, max_lng = metadata['Lng'].min(), metadata['Lng'].max()


    num_nodes = metadata.shape[0]
    
    feature_list = []

    # Initialize feature offset
    feat_offset = 0

    # Initialize config dictionary
    metadata_config = {}

    if add_location:
        locations = metadata[['Lat', 'Lng']].values.reshape(num_nodes, 2)
        # Normalize latitude and longitude
        locations = (locations - np.array([min_lat, min_lng])) / np.array([max_lat - min_lat, max_lng - min_lng])

        metadata_config['location'] = {
            'min_lat': min_lat,
            'max_lat': max_lat,
            'min_lng': min_lng,
            'max_lng': max_lng,
            'feature_num': 2,
            'feature_offset': feat_offset,
            'feature_names': ['Lat', 'Lng'],
            'feature_type': 'continuous',
        }
        feature_list.append(locations)
        feat_offset += 2
    
    if add_road:
        one_hot_road = pd.get_dummies(metadata['Fwy'], prefix='Road').values.reshape(num_nodes, -1)
        
        metadata_config['road'] = {
            'feature_num': one_hot_road.shape[1],
            'feature_offset': feat_offset,
            'feature_names': ['Fwy'],
            'feature_type': 'categorical',
        }

        feature_list.append(one_hot_road)
        feat_offset += one_hot_road.shape[1]

    if add_region:
        one_hot_region = pd.get_dummies(metadata[['District', 'County']], prefix='Region').values.reshape(num_nodes, -1)
        metadata_config['region'] = {
            'feature_num': one_hot_region.shape[1],
            'feature_offset': feat_offset,
            'feature_names': ['District', 'County'],
            'feature_type': 'categorical',
        }
        feature_list.append(one_hot_region)
        feat_offset += one_hot_region.shape[1]

    if add_lanes:
        lanes = metadata['Lanes'].values.reshape(num_nodes, 1)
        lanes = lanes / 8
        metadata_config['lanes'] = {
            'feature_num': 1,
            'feature_offset': feat_offset,
            'feature_names': ['Lanes'],
            'feature_type': 'continuous',
            'max_lanes': 8,
        }
        feature_list.append(lanes)
        feat_offset += 1

    if add_direction:
        one_hot_direction = pd.get_dummies(metadata['Direction'], prefix='Direction').values.reshape(num_nodes, -1)
        metadata_config['direction'] = {
            'feature_num': one_hot_direction.shape[1],
            'feature_offset': feat_offset,
            'feature_names': ['Direction'],
            'feature_type': 'categorical',
        }
        feature_list.append(one_hot_direction)
        feat_offset += one_hot_direction.shape[1]

    data = np.concatenate(feature_list, axis=-1).astype(float)
    
    return data, metadata_config

def to_adj_matrix(graph: nx.Graph, sensors: Optional[list], self_loops: bool = True) -> np.ndarray:
    """
    Converts the given GO-MO graph into a numpy adjacency matrix for the specified sensors.
    Self-loops are added to nodes corresponding to the sensors, and the adjacency matrix of the resulting graph is returned.

    The adjacency matrix is constructed based on the nodes specified in the sensors parameter.
    Only the subgraph consisting of these nodes is considered, inclusive of any weights on the edges.
    If sensors is None, the entire graph is considered.
    Self-loops will be added to these nodes.

    :param graph: The input GO-MO graph.
    :type graph: nx.Graph
    :param sensors: A list of nodes representing the subset of the graph for which the adjacency
        matrix is computed. If None, the entire graph is returned.
    :type sensors: list
    :param self_loops: If True, add self-loops (identity) to the adjacency matrix diagonal for the
        selected nodes.
    :type self_loops: bool
    :return: A numpy ndarray representing the weighted adjacency matrix of the largest
        connected subgraph corresponding to the nodes (sensors) provided.
    :rtype: np.ndarray
    """
    sensors = sensors or list(graph.nodes)

    # get adj matrix
    adj_matrix = nx.to_numpy_array(graph, nodelist=sensors, weight='weight')

    # add self loops if requested
    if self_loops:
        adj_matrix += np.eye(len(sensors))

    return adj_matrix


def generate_data_and_idx(df, x_offsets, y_offsets, add_time_of_day, add_day_of_week):
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    
    feature_list = [data]
    if add_time_of_day:
        time_ind = (df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
        time_of_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_of_day)
    if add_day_of_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        day_of_week = dow_tiled / 7
        feature_list.append(day_of_week)

    data = np.concatenate(feature_list, axis=-1)
    
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    print('idx min & max:', min_t, max_t)
    idx = np.arange(min_t, max_t, 1)
    return data, idx

def load_metadata(args):
    metadata = pd.read_csv(args.dataset + '/' + args.dataset + '_meta.csv')
    metadata = metadata.set_index('ID')
    return metadata

def new_and_dying_sensors(df: pd.DataFrame) -> tuple[list, list, list]:
    df = df.sort_index()

    isna   = df.isna()
    valid  = ~isna

    has_before = valid.cumsum(axis=0).gt(0)
    has_after  = valid[::-1].cumsum(axis=0)[::-1].gt(0)

    leading_nan = isna & ~has_before
    trailing_nan = isna & ~has_after
    internal_nan = isna & has_before & has_after

    newborn_cols = leading_nan.any(axis=0)
    dying_cols   = trailing_nan.any(axis=0)
    invalid_cols = internal_nan.any(axis=0) | isna.all(axis=0)

    newborn_sensors = df.columns[newborn_cols].tolist()
    dying_sensors   = df.columns[dying_cols].tolist()
    invalid_sensors = df.columns[invalid_cols].tolist()

    return newborn_sensors, dying_sensors, invalid_sensors


def generate_train_val_test(args):
    years = args.years.split('_')
    df = pd.DataFrame()
    for y in years:
        df_tmp = pd.read_hdf(args.dataset + '/' + args.dataset + '_his_' + y + '.h5')
        df = pd.concat([df, df_tmp])#df.append(df_tmp)
    print('original data shape:', df.shape)

    if args.dataset == 'mad':
        newborn, dying, temporary_off = new_and_dying_sensors(df)

        if len(temporary_off) > 0:
            print(f"Temporary off sensors found: {temporary_off}")

        to_drop = set(newborn) | set(dying) | set(temporary_off)
        df = df.drop(columns=to_drop)
        
        with open('mad/routes-graph.pkl', 'rb') as f:
            routes_network = pckl.load(f)
            
        mad_adjacency_matrix = to_adj_matrix(routes_network, sensors=df.columns.tolist(), self_loops=True)    
        
        np.save(args.dataset + '/' + args.dataset + '_rn_adj.npy', mad_adjacency_matrix)
        
        meta_df = pd.read_csv(args.dataset + '/' + args.dataset + '_meta.csv').set_index('ID')
        meta_df = meta_df[meta_df.index.isin(df.columns)]
        meta_df.to_csv(args.dataset + '/' + args.dataset + '_meta.csv', index=True)
        
    metadata_raw = load_metadata(args)
    metadata, metadata_config = generate_metadata(metadata_raw, True, True, True, True, False)

    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    x_offsets = np.arange(-(seq_length_x - 1), 1, 1)
    y_offsets = np.arange(1, (seq_length_y + 1), 1)

    data, idx = generate_data_and_idx(df, x_offsets, y_offsets, args.tod, args.dow)
    print('final data shape:', data.shape, 'idx shape:', idx.shape, 'metadata featues', metadata.shape[-1])

    num_samples = len(idx)
    num_train = round(num_samples * 0.6)
    num_val = round(num_samples * 0.2)   

    # split idx
    idx_train = idx[:num_train]
    idx_val = idx[num_train: num_train + num_val]
    idx_test = idx[num_train + num_val:]

    # normalize
    x_train = data[:idx_val[0] - args.seq_length_x, :, 0] 
    scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())
    data[..., 0] = scaler.transform(data[..., 0])

    # save
    out_dir = args.dataset + '/' + args.years
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.savez_compressed(os.path.join(out_dir, 'his.npz'), data=data, mean=scaler.mean, std=scaler.std, metadata = metadata, metadata_dict=metadata_config)

    np.save(os.path.join(out_dir, 'idx_train'), idx_train)
    np.save(os.path.join(out_dir, 'idx_val'), idx_val)
    np.save(os.path.join(out_dir, 'idx_test'), idx_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mad', help='dataset name')
    parser.add_argument('--years', type=str, default='2019', help='if use data from multiple years, please use underline to separate them, e.g., 2018_2019')
    parser.add_argument('--seq_length_x', type=int, default=12, help='sequence Length')
    parser.add_argument('--seq_length_y', type=int, default=12, help='sequence Length')
    parser.add_argument('--tod', type=int, default=1, help='time of day')
    parser.add_argument('--dow', type=int, default=1, help='day of week')
    
    args = parser.parse_args()
    generate_train_val_test(args)
