import os
import pickle
import torch
import json
import numpy as np
import threading
import multiprocessing as mp
from pathlib import Path

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
                num_threads = max(len(idx_ind) // 2, 1)
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

    for cat in ['train', 'val', 'test']:
        idx = np.load(os.path.join(data_path, args.years, 'idx_' + cat + '.npy'))

        if cat == 'train' and args.train_data_percentage < 1.0:
            num_train_samples = int(len(idx) * args.train_data_percentage)
            idx = idx[-num_train_samples:]
            logger.info(f'Using {args.train_data_percentage*100:.1f}% of the training data')

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
         'MAD': [base_dir+'mad', base_dir+'mad/mad_rn_adj.npy', 3965],
        }
    assert dataset in d.keys()
    return d[dataset]