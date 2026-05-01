import os
import pickle
import torch
import json
import numpy as np
import threading
import multiprocessing as mp
from pathlib import Path


import torch
from tqdm import tqdm

try:
    from fastdtw import fastdtw
except ImportError:
    fastdtw = None

try:
    from tslearn.clustering import TimeSeriesKMeans, KShape
except ImportError:
    TimeSeriesKMeans = None
    KShape = None

try:
    from scipy.sparse.csgraph import shortest_path
except ImportError:
    shortest_path = None

class DataLoader(object):
    def __init__(self, data, idx, seq_len, horizon, bs, logger, pad_last_sample=False, metadata = None, metadata_dict = None, input_mask = None, output_mask = None, available_sensors = None, drop_unavailable_sensors = False):
        """

        ## Parameters
        available_sensors: list or None, if None all sensors will be available. The shape should be (num_sensors, 1)

        """
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)
        
        self.data = data
        self.metadata = metadata
        self.metadata_dict = metadata_dict
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(self.size // self.bs)
        self.current_ind = 0
        logger.info('Sample num: ' + str(self.idx.shape[0]) + ', Batch num: ' + str(self.num_batch))
        
        self.available_sensors = available_sensors
        self.drop_unavailable_sensors = drop_unavailable_sensors
        
        self.input_mask = input_mask
        self.output_mask = output_mask

        if self.drop_unavailable_sensors and self.available_sensors is not None:
            logger.info('Dropping unavailable sensors from the input and output data.')
            self.data = self.data[:, self.available_sensors.squeeze() == 1]
            if self.input_mask is not None:
                self.input_mask = self.input_mask[..., self.available_sensors.squeeze() == 1]
            if self.output_mask is not None:
                self.output_mask = self.output_mask[..., self.available_sensors.squeeze() == 1] 
            if self.metadata is not None:
                self.metadata = self.metadata[self.available_sensors.squeeze() == 1]


        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon


        self.n_sensors = self.data.shape[1]

        assert self.input_mask is None or self.input_mask.shape[:2] == self.data.shape[:2]
        assert self.output_mask is None or self.output_mask.shape[:2] == self.data.shape[:2]

        self.data_in = self.data
        self.data_out = self.data.copy()
        # self.output_mask_value = -np.inf 
        self.output_mask_value = np.nan

        if self.input_mask is not None:
            self.input_mask = self.input_mask[..., np.newaxis]

        if self.output_mask is not None:
            self.output_mask = self.output_mask[..., np.newaxis]



    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx


    def write_to_shared_array(self, x, y, x_mask, y_mask, idx_ind, start_idx, end_idx):
        for i in range(start_idx, end_idx):
            tmp = self.data_in[idx_ind[i] + self.x_offsets, :, :]
            
            ## The input mask has to be applied before the metadata is concatenated
            if self.input_mask is not None:
                tmp = tmp * self.input_mask[idx_ind[i] + self.x_offsets, :, :]
                x_mask[i] = self.input_mask[idx_ind[i] + self.x_offsets, :, :]
            else :
                x_mask[i] = 1


            ## Concatenate metadata if available
            if self.metadata is not None:
                tmp = np.concatenate([
                    tmp, 
                    np.tile(self.metadata, (self.seq_len, 1, 1))
                ], axis = -1)

            y_tmp = self.data_out[idx_ind[i] + self.y_offsets, :, :1]
            if self.output_mask is not None:
                y_tmp = np.where(self.output_mask[idx_ind[i] + self.y_offsets, :, :] == 0, self.output_mask_value, y_tmp)
                y_mask[i] = self.output_mask[idx_ind[i] + self.y_offsets, :, :]
            else:
                y_mask[i] = 1


            ## Remove the unavailable sensors at the end to zero out all their values
            if self.available_sensors is not None and not self.drop_unavailable_sensors:
                tmp = tmp * self.available_sensors[np.newaxis, :, :]
                x_mask[i] = x_mask[i] * self.available_sensors[np.newaxis, :, :]

                y_tmp = y_tmp * self.available_sensors[np.newaxis, :, :]
                y_mask[i] = y_mask[i] * self.available_sensors[np.newaxis, :, :]
            
            x[i] = tmp
            y[i] = y_tmp


    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind: end_ind, ...]

                x_shape = (len(idx_ind), self.seq_len, self.data.shape[1], self.data.shape[-1] if self.metadata is None else self.data.shape[-1] + self.metadata.shape[-1])
                x_shared = mp.RawArray('f', int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype='f').reshape(x_shape)

                x_mask_shape = (len(idx_ind), self.seq_len, self.data.shape[1], 1)
                x_mask_shared = mp.RawArray('b', int(np.prod(x_mask_shape)))
                x_mask = np.frombuffer(x_mask_shared, dtype='b').reshape(x_mask_shape)

                y_shape = (len(idx_ind), self.horizon, self.data.shape[1], 1)
                y_shared = mp.RawArray('f', int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype='f').reshape(y_shape)

                y_mask_shape = (len(idx_ind), self.horizon, self.data.shape[1], 1)
                y_mask_shared = mp.RawArray('b', int(np.prod(y_mask_shape)))
                y_mask = np.frombuffer(y_mask_shared, dtype='b').reshape(y_mask_shape)

                array_size = len(idx_ind)
                num_threads = len(idx_ind) // 2
                chunk_size = array_size // num_threads
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = start_index + chunk_size if i < num_threads - 1 else array_size
                    thread = threading.Thread(target=self.write_to_shared_array, args=(x, y, x_mask, y_mask, idx_ind, start_index, end_index))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield (x, y, x_mask, y_mask)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)


    def transform(self, data):
        return (data - self.mean) / self.std


    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(data_path, args, logger, drop_unavailable_sensors = False):
    ptr = np.load(os.path.join(data_path, args.years, 'his.npz'), allow_pickle=True)
    logger.info('Data shape: ' + str(ptr['data'].shape))
    
    dataloader = {}

    use_masks = args.mask_name is not None

    if use_masks:
        mask_path = Path('data/masks') / args.dataset.lower() / args.mask_name / f'mask_{args.mask_iter:02d}'
        input_mask = torch.load(mask_path / 'input_mask.pt').numpy().squeeze()
        output_mask = torch.load(mask_path / 'output_mask.pt').numpy().squeeze()

        mask_config = json.load(open(mask_path / '../config.json', 'r'))

        if mask_config.get('train_dropout', 0) > 0:
            train_mask = torch.load(mask_path / 'train_mask.pt').numpy().squeeze()
            # train_mask = np.tile( train_mask[np.newaxis, :], (input_mask.shape[0], 1))
            train_mask = train_mask.reshape(-1, 1)
        else:
            train_mask = None
    else:
        input_mask = None
        output_mask = None
        train_mask = None

    if args.use_metadata:
            metadata = ptr.get('metadata', None)
            metadata_dict = ptr.get('metadata_dict', None)
    else:
        metadata = None
        metadata_dict = None

    idx_dict = {}
    for cat in ['train', 'val', 'test']:
        idx = np.load(os.path.join(data_path, args.years, 'idx_' + cat + '.npy'))
        idx_dict[cat] = idx

        if use_masks and (cat == 'train' or cat == 'val'):
            dataloader[cat + '_loader'] = DataLoader(ptr['data'][..., :args.input_dim], idx, \
                                                 args.seq_len, args.horizon, args.bs, logger, 
                                                 metadata = metadata, 
                                                 metadata_dict = metadata_dict, 
                                                 input_mask=input_mask,
                                                 output_mask=output_mask,
                                                 available_sensors = train_mask,
                                                 drop_unavailable_sensors = drop_unavailable_sensors,
                                                 )
        else:         
            dataloader[cat + '_loader'] = DataLoader(ptr['data'][..., :args.input_dim], idx, \
                                                 args.seq_len, args.horizon, args.bs, logger, 
                                                 metadata = metadata, 
                                                 metadata_dict = metadata_dict, 
                                                 input_mask=input_mask,
                                                 output_mask=output_mask,
                                                 available_sensors = None
                                                 )
            if cat == 'test':
                dataloader['benchmark_loader'] = DataLoader(ptr['data'][..., :args.input_dim], idx, \
                                                    args.seq_len, args.horizon, args.bs, logger, 
                                                    metadata = metadata, 
                                                    metadata_dict = metadata_dict, 
                                                    input_mask=input_mask,
                                                    output_mask=output_mask,
                                                    available_sensors = None
                                                    )

    scaler = StandardScaler(mean=ptr['mean'], std=ptr['std'])

    if getattr(args, "model_description", "").lower() == "pdformer":
        data_feature = build_pdformer_data_feature(
            raw_data=ptr['data'],
            train_idx=idx_dict['train'],
            scaler=scaler,
            args=args,
            logger=logger,
            metadata=metadata,
        )
        data_feature["num_batches"] = dataloader["train_loader"].num_batch

        return dataloader, scaler, data_feature
    
    else:    
        return dataloader, scaler


def load_adj_from_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj_from_numpy(numpy_file):
    return np.load(numpy_file)


def get_dataset_info(dataset):
    base_dir = os.getcwd() + '/data/'
    d = {
         'CA': [base_dir+'ca', base_dir+'ca/ca_rn_adj.npy', 8600],
         'GLA': [base_dir+'gla', base_dir+'gla/gla_rn_adj.npy', 3834],
         'GBA': [base_dir+'gba', base_dir+'gba/gba_rn_adj.npy', 2352],
         'SD': [base_dir+'sd', base_dir+'sd/sd_rn_adj.npy', 716],
        }
    assert dataset in d.keys()
    return d[dataset]







# PDFormer helper functions
def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _extract_daily_average(data, points_per_day, output_dim=1):
    """
    data shape: (T, N, F)
    returns: (points_per_day, N, output_dim)
    """
    total_days = data.shape[0] // points_per_day
    usable = total_days * points_per_day
    data = data[:usable, :, :output_dim]
    data = data.reshape(total_days, points_per_day, data.shape[1], output_dim)
    return data.mean(axis=0)


def compute_dtw_matrix(data, dataset_name, years, time_intervals, output_dim, cache_dir, logger):
    """
    data shape: (T, N, F), should be the canonical raw tensor before metadata concatenation.
    """
    if fastdtw is None:
        raise ImportError("fastdtw is required for PDFormer DTW matrix computation.")

    _ensure_dir(cache_dir)
    cache_path = os.path.join(cache_dir, f"dtw_{dataset_name}_{years}_odim{output_dim}_ti{time_intervals}.npy")

    if os.path.exists(cache_path):
        dtw_matrix = np.load(cache_path)
        logger.info(f"Loaded DTW matrix from {cache_path}")
        return dtw_matrix

    points_per_day = 24 * 3600 // time_intervals
    data_mean = _extract_daily_average(data, points_per_day, output_dim=output_dim)

    num_nodes = data.shape[1]
    dtw_distance = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    logger.info("Computing DTW matrix for PDFormer...")
    for i in tqdm(range(num_nodes), desc="Computing DTW matrix"):
        for j in range(i, num_nodes):
            dist, _ = fastdtw(data_mean[:, i, :], data_mean[:, j, :], radius=6)
            dtw_distance[i, j] = dist
            dtw_distance[j, i] = dist

    np.save(cache_path, dtw_distance)
    logger.info(f"Saved DTW matrix to {cache_path}")
    return dtw_distance


def materialize_x_from_idx(data, idx, seq_len, metadata=None):
    """
    Build x windows only.
    data shape: (T, N, F)
    idx shape: (num_samples,)
    returns x shape: (num_samples, seq_len, N, F [+ metadata_dim])
    """
    x_offsets = np.arange(-(seq_len - 1), 1, 1)
    xs = []
    for i in idx:
        x = data[i + x_offsets, :, :]
        if metadata is not None:
            x = np.concatenate(
                [x, np.tile(metadata, (seq_len, 1, 1))],
                axis=-1
            )
        xs.append(x)
    return np.stack(xs, axis=0)


def compute_pattern_keys(
    x_train,
    dataset_name,
    years,
    cand_key_days,
    s_attn_size,
    n_cluster,
    cluster_max_iter,
    cluster_method,
    time_intervals,
    output_dim,
    cache_dir,
    logger,
):
    """
    x_train shape: (num_samples, seq_len, N, F)
    Only first output_dim channels are used for PDFormer pattern keys.
    """
    _ensure_dir(cache_dir)
    cache_path = os.path.join(
        cache_dir,
        f"pattern_keys_{cluster_method}_{dataset_name}_{years}_days{cand_key_days}_"
        f"s{s_attn_size}_k{n_cluster}_iter{cluster_max_iter}_odim{output_dim}_ti{time_intervals}.npy"
    )

    if os.path.exists(cache_path):
        pattern_keys = np.load(cache_path)
        logger.info(f"Loaded pattern keys from {cache_path}")
        return pattern_keys

    points_per_day = 24 * 3600 // time_intervals
    cand_key_time_steps = cand_key_days * points_per_day
    cand_key_time_steps = min(cand_key_time_steps, x_train.shape[0])
    pattern_cand_keys = x_train[:cand_key_time_steps, :s_attn_size, :, :output_dim]

    pattern_cand_keys = pattern_cand_keys.swapaxes(1, 2).reshape(-1, s_attn_size, output_dim)

    logger.info(f"Clustering pattern keys for PDFormer with method={cluster_method}...")
    if cluster_method.lower() == "kshape":
        if KShape is None:
            raise ImportError("tslearn is required for KShape clustering.")
        km = KShape(n_clusters=n_cluster, max_iter=cluster_max_iter, random_state=0).fit(pattern_cand_keys)
    else:
        if TimeSeriesKMeans is None:
            raise ImportError("tslearn is required for TimeSeriesKMeans clustering.")
        km = TimeSeriesKMeans(
            n_clusters=n_cluster,
            metric="softdtw",
            max_iter=cluster_max_iter,
            random_state=0,
        ).fit(pattern_cand_keys)

    pattern_keys = km.cluster_centers_.astype(np.float32)
    np.save(cache_path, pattern_keys)
    logger.info(f"Saved pattern keys to {cache_path}")
    return pattern_keys


def compute_pdformer_graph_matrices(dist_mx, type_short_path="hop", weight_adj_epsilon=0.1, logger=None):
    """
    dist_mx:
      raw road-network distance matrix
      shape (N, N), with inf for disconnected if applicable

    returns:
      adj_mx, sd_mx, sh_mx
    """
    dist_mx = dist_mx.copy().astype(np.float32)
    N = dist_mx.shape[0]

    # Gaussian kernel adjacency
    finite_mask = np.isfinite(dist_mx)
    finite_dist = dist_mx[finite_mask]
    std = finite_dist.std() if finite_dist.size > 0 else 1.0
    std = max(std, 1e-6)

    adj_mx = np.exp(-np.square(dist_mx / std)).astype(np.float32)
    adj_mx[~finite_mask] = 0.0
    adj_mx[adj_mx < weight_adj_epsilon] = 0.0

    sd_mx = None
    sh_mx = None

    # if type_short_path == "dist":
    #     sd_mx = dist_mx.copy()
    #     sd_mx[adj_mx == 0] = np.inf
    #     for k in range(N):
    #         sd_mx = np.minimum(sd_mx, sd_mx[:, [k]] + sd_mx[[k], :])

    if type_short_path == "dist":
        if shortest_path is None:
            raise ImportError("scipy is required for shortest-distance matrix computation.")
        sd_mx = dist_mx.copy()
        sd_mx[adj_mx == 0] = np.inf
        sd_mx = shortest_path(sd_mx, directed=False, unweighted=False).astype(np.float32)
        if logger:
            logger.info("Computed shortest-distance matrix for PDFormer.")

    # elif type_short_path == "hop":
    #     # TODO: Changed
    #     sh_mx = dist_mx.copy()
    #     sh_mx[sh_mx > 0] = 1
    #     sh_mx[~np.isfinite(sh_mx)] = 511
    #     sh_mx[sh_mx == 0] = 511
    #     np.fill_diagonal(sh_mx, 0)
    #     sh_mx = sh_mx.astype(np.int32)

    #     for k in range(N):
    #         sh_mx = np.minimum(sh_mx, sh_mx[:, [k]] + sh_mx[[k], :])
    #         sh_mx = np.minimum(sh_mx, 511)

    #     if logger:
    #         logger.info("Computed shortest-hop matrix for PDFormer.")
    elif type_short_path == "hop":
        if shortest_path is None:
            raise ImportError("scipy is required for shortest-hop matrix computation.")
        hop_mx = np.where(np.isfinite(dist_mx) & (dist_mx > 0), 1, np.inf).astype(np.float32)
        np.fill_diagonal(hop_mx, 0)
        sh_mx = shortest_path(hop_mx, directed=False, unweighted=True)
        sh_mx = np.minimum(sh_mx, 511).astype(np.int32)
        if logger:
            logger.info("Computed shortest-hop matrix for PDFormer.")

    return adj_mx, sd_mx, sh_mx


def build_pdformer_data_feature(
    raw_data,
    train_idx,
    scaler,
    args,
    logger,
    metadata=None,
):
    """
    raw_data shape: (T, N, F)
    """
    cache_dir = os.path.join("data", "cache", "pdformer")
    _ensure_dir(cache_dir)

    time_intervals = getattr(args, "time_intervals", 300)
    output_dim = getattr(args, "output_dim", 1)
    type_short_path = getattr(args, "type_short_path", "dist")
    weight_adj_epsilon = getattr(args, "weight_adj_epsilon", 0.1)
    cand_key_days = getattr(args, "cand_key_days", 21)
    s_attn_size = getattr(args, "s_attn_size", 3)
    n_cluster = getattr(args, "n_cluster", 16)
    cluster_max_iter = getattr(args, "cluster_max_iter", 5)
    cluster_method = getattr(args, "cluster_method", "kshape")

    dtw_matrix = compute_dtw_matrix(
        data=raw_data,
        dataset_name=args.dataset,
        years=args.years,
        time_intervals=time_intervals,
        output_dim=output_dim,
        cache_dir=cache_dir,
        logger=logger,
    )


    points_per_day = 24 * 3600 // time_intervals
    needed = min(len(train_idx), cand_key_days * points_per_day)

    x_train_for_keys = materialize_x_from_idx(
        data=raw_data[..., :args.input_dim],
        idx=train_idx[:needed],
        seq_len=args.seq_len,
        metadata=metadata if args.use_metadata else None,
    )
    # TODO: CHANGED
    # x_train_for_keys = materialize_x_from_idx(
    #     data=raw_data[..., :args.input_dim],
    #     idx=train_idx,
    #     seq_len=args.seq_len,
    #     metadata=metadata if args.use_metadata else None,
    # )

    pattern_keys = compute_pattern_keys(
        x_train=x_train_for_keys,
        dataset_name=args.dataset,
        years=args.years,
        cand_key_days=cand_key_days,
        s_attn_size=s_attn_size,
        n_cluster=n_cluster,
        cluster_max_iter=cluster_max_iter,
        cluster_method=cluster_method,
        time_intervals=time_intervals,
        output_dim=output_dim,
        cache_dir=cache_dir,
        logger=logger,
    )

    _, adj_path, num_nodes = get_dataset_info(args.dataset)
    dist_mx = load_adj_from_numpy(adj_path)

    adj_mx, sd_mx, sh_mx = compute_pdformer_graph_matrices(
        dist_mx=dist_mx,
        type_short_path=type_short_path,
        weight_adj_epsilon=weight_adj_epsilon,
        logger=logger,
    )

    feature_dim = args.input_dim + (metadata.shape[-1] if (args.use_metadata and metadata is not None) else 0)
    ext_dim = feature_dim - output_dim

    return {
        "scaler": scaler,
        "adj_mx": adj_mx,
        "sd_mx": sd_mx,
        "sh_mx": sh_mx,
        "dtw_matrix": dtw_matrix,
        "pattern_keys": pattern_keys,
        "num_nodes": num_nodes,
        "feature_dim": feature_dim,
        "output_dim": output_dim,
        "ext_dim": ext_dim,
    }