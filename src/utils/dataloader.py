import os
import pickle
import torch
import json
import queue
import numpy as np
import threading
from pathlib import Path

class DataLoader(object):
    def __init__(self, data, idx, seq_len, horizon, bs, logger, pad_last_sample=False, metadata = None, metadata_dict = None, input_mask = None, output_mask = None, available_sensors = None, drop_unavailable_sensors = False, data_c0 = None, device = None, prefetch_depth = 2, dataset = None):
        """

        ## Parameters
        available_sensors: list or None, if None all sensors will be available. The shape should be (num_sensors, 1)
        data_c0: optional pre-built contiguous copy of data[..., :1], shared between loaders.
        device: torch device the assembled batches are uploaded to. Defaults to CPU.
        prefetch_depth: how many batches are assembled ahead of the consumer. 0 disables
            the background thread and builds every batch on demand.

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
            keep = self.available_sensors.squeeze() == 1
            # load_dataset applies the subset once and shares it between the train and
            # val loaders, so only do the work here when it is still pending.
            if self.data.shape[1] != int(keep.sum()):
                logger.info('Dropping unavailable sensors from the input and output data.')
                self.data = self.data[:, keep]
                data_c0 = None
                if self.input_mask is not None:
                    self.input_mask = self.input_mask[..., keep]
                if self.output_mask is not None:
                    self.output_mask = self.output_mask[..., keep]
                if self.metadata is not None:
                    self.metadata = self.metadata[keep]


        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon
        self.dataset = dataset


        self.n_sensors = self.data.shape[1]

        assert self.input_mask is None or self.input_mask.shape[:2] == self.data.shape[:2]
        assert self.output_mask is None or self.output_mask.shape[:2] == self.data.shape[:2]

        # self.output_mask_value = -np.inf
        self.output_mask_value = np.nan

        # The per-batch gather uses take(mode='clip'), because the default mode='raise'
        # takes a slow path that costs two orders of magnitude more (440ms vs 3.2ms per
        # batch on CA). Checking the window bounds once here is what makes clipping safe.
        assert self.idx.min() - (seq_len - 1) >= 0 and self.idx.max() + horizon < self.data.shape[0], \
            'idx contains windows that reach outside the data array'

        self.data = np.ascontiguousarray(self.data, dtype=np.float32)
        if data_c0 is not None and data_c0.shape[1] == self.n_sensors:
            self.data_c0 = data_c0
        else:
            # The label gather only reads feature 0. A contiguous copy of that column
            # avoids a three times wider strided gather on every batch.
            self.data_c0 = np.ascontiguousarray(self.data[..., :1])

        if self.input_mask is not None:
            self.input_mask = np.ascontiguousarray(
                self.input_mask, dtype=np.int8).reshape(-1, self.n_sensors, 1)

        if self.output_mask is not None:
            self.output_mask = np.ascontiguousarray(
                self.output_mask, dtype=np.int8).reshape(-1, self.n_sensors, 1)

        self.n_features = self.data.shape[-1]
        self.n_metadata = 0 if self.metadata is None else self.metadata.shape[-1]

        # Availability zeroes out whole sensors (metadata columns included) only when the
        # unavailable ones are kept in the tensor; dropping them makes it a no-op.
        self._scale_by_availability = self.available_sensors is not None and not self.drop_unavailable_sensors
        if self._scale_by_availability:
            # available_sensors stays full length even when the unavailable sensors are
            # dropped, so it only lines up with n_sensors when we are scaling.
            avail = np.asarray(self.available_sensors, dtype=np.float32).reshape(1, self.n_sensors, 1)
            self._avail_f32 = avail
            self._avail_i8 = avail.astype(np.int8)
        else:
            self._avail_f32 = None
            self._avail_i8 = None

        self._device = torch.device('cpu') if device is None else torch.device(device)
        self._prefetch_depth = max(int(prefetch_depth), 0)
        self._logger = logger
        self._slots = None
        self._slot_events = None
        self._copy_stream = None


    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx


    def _allocate_slots(self):
        """Allocate the ring of pinned host buffers that every batch is assembled into.

        A slot is refilled only after the host-to-device copy that read it has completed,
        so the ring needs prefetch_depth + 2 entries: one being filled, one in flight, and
        prefetch_depth sitting in the queue.
        """
        if self._slots is not None:
            return

        n_slots = self._prefetch_depth + 2 if self._prefetch_depth > 0 else 1
        pin = self._device.type == 'cuda'
        n, f, m = self.n_sensors, self.n_features, self.n_metadata

        def buffer(shape, dtype):
            try:
                return torch.empty(shape, dtype=dtype, pin_memory=pin)
            except RuntimeError:
                # Pinning can fail for very large batches (CA with metadata is ~5GB per
                # slot). Fall back to pageable memory rather than dying.
                self._logger.info('Could not pin loader buffers, falling back to pageable memory.')
                return torch.empty(shape, dtype=dtype)

        self._slots = []
        for _ in range(n_slots):
            x = buffer((self.bs, self.seq_len, n, f + m), torch.float32)
            y = buffer((self.bs, self.horizon, n, 1), torch.float32)
            x_mask = buffer((self.bs, self.seq_len, n, 1), torch.int8)
            y_mask = buffer((self.bs, self.horizon, n, 1), torch.int8)

            slot = {
                'tensors': (x, y, x_mask, y_mask),
                'x': x.numpy(),
                'y': y.numpy(),
                'x_mask': x_mask.numpy(),
                'y_mask': y_mask.numpy(),
                'stage': None,
            }
            if m > 0:
                # The metadata columns are identical for every sample of every batch, so
                # write them once here; only the data columns change per batch.
                metadata = self.metadata
                if self._scale_by_availability:
                    metadata = metadata * self._avail_f32.reshape(n, 1)
                slot['x'][..., f:] = metadata
                # x[..., :f] is strided once metadata is appended, so the gather needs a
                # contiguous staging array to take() into.
                slot['stage'] = np.empty((self.bs * self.seq_len, n, f), dtype=np.float32)
            self._slots.append(slot)

        self._slot_events = [None] * n_slots
        if pin and self._prefetch_depth > 0:
            self._copy_stream = torch.cuda.Stream(device=self._device)


    def _fill_slot(self, slot, idx_ind):
        """Assemble one batch into the host buffers of ``slot``."""
        b = self._slots[slot]
        n, f = self.n_sensors, self.n_features
        wx = (idx_ind[:, None] + self.x_offsets).ravel()
        wy = (idx_ind[:, None] + self.y_offsets).ravel()

        stage = b['stage'] if b['stage'] is not None else b['x'].reshape(-1, n, f)
        x_mask_flat = b['x_mask'].reshape(-1, n, 1)
        self.data.take(wx, axis=0, out=stage, mode='clip')

        ## The input mask has to be applied before the metadata is concatenated
        if self.input_mask is not None:
            self.input_mask.take(wx, axis=0, out=x_mask_flat, mode='clip')
            np.multiply(stage, x_mask_flat, out=stage)
        else:
            b['x_mask'].fill(1)

        ## Remove the unavailable sensors at the end to zero out all their values
        if self._scale_by_availability:
            np.multiply(stage, self._avail_f32, out=stage)
            np.multiply(b['x_mask'], self._avail_i8, out=b['x_mask'])

        if b['stage'] is not None:
            b['x'][..., :f] = stage.reshape(self.bs, self.seq_len, n, f)

        y_flat = b['y'].reshape(-1, n, 1)
        y_mask_flat = b['y_mask'].reshape(-1, n, 1)
        self.data_c0.take(wy, axis=0, out=y_flat, mode='clip')

        if self.output_mask is not None:
            self.output_mask.take(wy, axis=0, out=y_mask_flat, mode='clip')
            y_flat[y_mask_flat == 0] = self.output_mask_value
        else:
            b['y_mask'].fill(1)

        if self._scale_by_availability:
            np.multiply(b['y'], self._avail_f32, out=b['y'])
            np.multiply(b['y_mask'], self._avail_i8, out=b['y_mask'])


    def _upload(self, slot):
        """Move a filled slot to the device. Returns (tensors, copy-finished event)."""
        tensors = self._slots[slot]['tensors']

        if self._copy_stream is None:
            if self._device.type == 'cpu':
                # Nothing to copy, but the consumer must not see a buffer we are about
                # to refill, so hand out a detached copy.
                return tuple(t.clone() for t in tensors), None
            return tuple(t.to(self._device) for t in tensors), None

        with torch.cuda.stream(self._copy_stream):
            device_tensors = tuple(t.to(self._device, non_blocking=True) for t in tensors)
        event = torch.cuda.Event()
        event.record(self._copy_stream)
        self._slot_events[slot] = event
        return device_tensors, event


    def _consume(self, batch):
        """Order the compute stream behind the upload before handing the batch out."""
        device_tensors, event = batch
        if event is None:
            return device_tensors
        stream = torch.cuda.current_stream(self._device)
        stream.wait_event(event)
        for tensor in device_tensors:
            # These were allocated on the copy stream; tell the caching allocator they
            # are now in use by the compute stream before it can hand them to anyone else.
            tensor.record_stream(stream)
        return device_tensors


    def _batch_idx(self, batch):
        start_ind = self.bs * batch
        end_ind = min(self.size, self.bs * (batch + 1))
        return self.idx[start_ind: end_ind, ...]


    def get_iterator(self):
        self.current_ind = 0
        self._allocate_slots()

        if self._prefetch_depth == 0:
            return self._sync_iterator()
        return self._prefetch_iterator()


    def _sync_iterator(self):
        def _wrapper():
            while self.current_ind < self.num_batch:
                self._fill_slot(0, self._batch_idx(self.current_ind))
                yield self._consume(self._upload(0))
                self.current_ind += 1

        return _wrapper()


    def _prefetch_iterator(self):
        """Assemble batches on a background thread so that the gather and the
        host-to-device copy overlap with the training step of the previous batch."""
        stop = threading.Event()
        out_q = queue.Queue(maxsize=self._prefetch_depth)
        done = object()

        def produce():
            try:
                for batch in range(self.num_batch):
                    if stop.is_set():
                        break
                    slot = batch % len(self._slots)
                    event = self._slot_events[slot]
                    if event is not None:
                        # Wait for the copy that read this slot before overwriting it.
                        event.synchronize()
                    self._fill_slot(slot, self._batch_idx(batch))
                    out_q.put(self._upload(slot))
            except BaseException as exc:  # re-raised on the consumer side
                out_q.put(exc)
            else:
                out_q.put(done)

        thread = threading.Thread(target=produce, daemon=True, name='dataloader-prefetch')
        thread.start()

        def _wrapper():
            try:
                while True:
                    item = out_q.get()
                    if item is done:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    yield self._consume(item)
                    self.current_ind += 1
            finally:
                # Callers routinely abandon the iterator (an early break, or taking a
                # single batch), so drain the queue to unblock a producer stuck in put().
                stop.set()
                while thread.is_alive():
                    try:
                        out_q.get(timeout=0.05)
                    except queue.Empty:
                        pass
                thread.join()

        return _wrapper()


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)


    def transform(self, data):
        return (data - self.mean) / self.std


    def inverse_transform(self, data):
        return (data * self.std) + self.mean

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


def _load_source_f32(data_path, years, logger):
    """Return the (time, sensor, feature) history as a memory-mapped float32 array.

    The npz holds float64 and has to be decompressed on every access (14.6s for CA),
    which load_dataset used to pay once per split. Decoding it to a plain .npy once and
    mapping that keeps all splits on a single page-cached copy at half the size.
    """
    npz_path = os.path.join(data_path, years, 'his.npz')
    cache_dir = Path(data_path).parent / 'cache'
    cache_path = cache_dir / '{}_{}_f32.npy'.format(Path(data_path).name, years)

    if not cache_path.exists() or cache_path.stat().st_mtime < os.path.getmtime(npz_path):
        logger.info('Building float32 source cache at ' + str(cache_path))
        cache_dir.mkdir(parents=True, exist_ok=True)
        with np.load(npz_path, allow_pickle=True) as ptr:
            data = ptr['data'].astype(np.float32)
        # Write via a temporary file so a crash or a concurrent run never leaves a
        # truncated cache behind. np.save is given a handle so it does not append .npy.
        tmp_path = cache_path.with_name(cache_path.name + '.tmp{}'.format(os.getpid()))
        try:
            with open(tmp_path, 'wb') as handle:
                np.save(handle, data)
            os.replace(tmp_path, cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        del data

    return np.load(cache_path, mmap_mode='r')


def load_dataset(data_path, args, logger, drop_unavailable_sensors = False):
    source = _load_source_f32(data_path, args.years, logger)
    logger.info('Data shape: ' + str(source.shape))

    data = source[..., :args.input_dim]
    if not data.flags['C_CONTIGUOUS']:
        # take(axis=0, out=...) needs a contiguous source to stay on its fast path.
        data = np.ascontiguousarray(data)
    data_c0 = np.ascontiguousarray(data[..., :1])

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
        with np.load(os.path.join(data_path, args.years, 'his.npz'), allow_pickle=True) as ptr:
            metadata = ptr.get('metadata', None)
            metadata_dict = ptr.get('metadata_dict', None)
        if metadata is not None:
            metadata = np.ascontiguousarray(metadata, dtype=np.float32)
    else:
        metadata = None
        metadata_dict = None

    full = dict(data=data, data_c0=data_c0, metadata=metadata,
                input_mask=input_mask, output_mask=output_mask)

    # The train and val loaders drop the same sensors, so do the subset once and share it.
    subset = None
    if drop_unavailable_sensors and train_mask is not None:
        keep = train_mask.squeeze() == 1
        subset = dict(
            data=np.ascontiguousarray(data[:, keep]),
            metadata=None if metadata is None else np.ascontiguousarray(metadata[keep]),
            input_mask=None if input_mask is None else np.ascontiguousarray(input_mask[..., keep]),
            output_mask=None if output_mask is None else np.ascontiguousarray(output_mask[..., keep]),
        )
        subset['data_c0'] = np.ascontiguousarray(subset['data'][..., :1])

    device = torch.device(args.device) if getattr(args, 'device', '') else torch.device('cpu')
    prefetch_depth = getattr(args, 'prefetch_depth', 2)

    def build(arrays, idx, available_sensors, drop):
        return DataLoader(arrays['data'], idx, args.seq_len, args.horizon, args.bs, logger,
                          metadata = arrays['metadata'],
                          metadata_dict = metadata_dict,
                          input_mask = arrays['input_mask'],
                          output_mask = arrays['output_mask'],
                          available_sensors = available_sensors,
                          drop_unavailable_sensors = drop,
                          data_c0 = arrays['data_c0'],
                          device = device,
                          prefetch_depth = prefetch_depth,
                          dataset = args.dataset.upper() 
                          )
    
    for cat in ['train', 'val', 'test']:
        idx = np.load(os.path.join(data_path, args.years, 'idx_' + cat + '.npy'))

        if cat == 'train' and args.train_data_percentage < 1.0:
            num_train_samples = int(len(idx) * args.train_data_percentage)
            idx = idx[-num_train_samples:]
            logger.info(f'Using {args.train_data_percentage*100:.1f}% of the training data')

        if use_masks and (cat == 'train' or cat == 'val'):
            dataloader[cat + '_loader'] = build(subset if subset is not None else full, idx,
                                                train_mask, drop_unavailable_sensors)
        else:
            dataloader[cat + '_loader'] = build(full, idx, None, False)
            if train_mask is not None and not drop_unavailable_sensors:
                dataloader[cat + '_loader_drop'] = build(subset if subset is not None else full, idx, train_mask, False)
            if cat == 'test':
                dataloader['benchmark_loader'] = build(full, idx, None, False)

    with np.load(os.path.join(data_path, args.years, 'his.npz'), allow_pickle=True) as ptr:
        scaler = StandardScaler(mean=ptr['mean'][()], std=ptr['std'][()])
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