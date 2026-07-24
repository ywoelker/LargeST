"""
Equivalence tests for the rewritten src/utils/dataloader.DataLoader.

The loader used to assemble each batch with a per-sample Python loop writing into
freshly allocated multiprocessing.RawArray buffers. That loop is reproduced verbatim
below as ReferenceDataLoader, and every test asserts the vectorised implementation
produces the same four arrays for the same indices, across the full matrix of
metadata / input mask / output mask / available-sensors options.

Run with: python tests/test_dataloader_equivalence.py
"""

import logging
import os
import sys
import threading

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.dataloader import DataLoader


LOGGER = logging.getLogger("test_dataloader")
LOGGER.addHandler(logging.NullHandler())

SEQ_LEN = 12
HORIZON = 12


class ReferenceDataLoader:
    """The pre-optimisation batch assembly, kept as the oracle for these tests."""

    def __init__(self, data, idx, seq_len, horizon, bs, metadata=None, input_mask=None,
                 output_mask=None, available_sensors=None, drop_unavailable_sensors=False):
        self.data = data
        self.metadata = metadata
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(self.size // self.bs)
        self.available_sensors = available_sensors
        self.drop_unavailable_sensors = drop_unavailable_sensors
        self.input_mask = input_mask
        self.output_mask = output_mask

        if self.drop_unavailable_sensors and self.available_sensors is not None:
            keep = self.available_sensors.squeeze() == 1
            self.data = self.data[:, keep]
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
        self.data_in = self.data
        self.data_out = self.data.copy()
        self.output_mask_value = np.nan

        if self.input_mask is not None:
            self.input_mask = self.input_mask[..., np.newaxis]
        if self.output_mask is not None:
            self.output_mask = self.output_mask[..., np.newaxis]

    def get_iterator(self):
        for batch in range(self.num_batch):
            idx_ind = self.idx[self.bs * batch: self.bs * (batch + 1)]
            n = self.data.shape[1]
            n_feat = self.data.shape[-1] if self.metadata is None \
                else self.data.shape[-1] + self.metadata.shape[-1]

            x = np.empty((len(idx_ind), self.seq_len, n, n_feat), dtype=np.float32)
            x_mask = np.empty((len(idx_ind), self.seq_len, n, 1), dtype=np.int8)
            y = np.empty((len(idx_ind), self.horizon, n, 1), dtype=np.float32)
            y_mask = np.empty((len(idx_ind), self.horizon, n, 1), dtype=np.int8)

            for i in range(len(idx_ind)):
                tmp = self.data_in[idx_ind[i] + self.x_offsets, :, :]

                if self.input_mask is not None:
                    tmp = tmp * self.input_mask[idx_ind[i] + self.x_offsets, :, :]
                    x_mask[i] = self.input_mask[idx_ind[i] + self.x_offsets, :, :]
                else:
                    x_mask[i] = 1

                if self.metadata is not None:
                    tmp = np.concatenate([
                        tmp,
                        np.tile(self.metadata, (self.seq_len, 1, 1))
                    ], axis=-1)

                y_tmp = self.data_out[idx_ind[i] + self.y_offsets, :, :1]
                if self.output_mask is not None:
                    y_tmp = np.where(self.output_mask[idx_ind[i] + self.y_offsets, :, :] == 0,
                                     self.output_mask_value, y_tmp)
                    y_mask[i] = self.output_mask[idx_ind[i] + self.y_offsets, :, :]
                else:
                    y_mask[i] = 1

                if self.available_sensors is not None and not self.drop_unavailable_sensors:
                    tmp = tmp * self.available_sensors[np.newaxis, :, :]
                    x_mask[i] = x_mask[i] * self.available_sensors[np.newaxis, :, :]
                    y_tmp = y_tmp * self.available_sensors[np.newaxis, :, :]
                    y_mask[i] = y_mask[i] * self.available_sensors[np.newaxis, :, :]

                x[i] = tmp
                y[i] = y_tmp

            yield (x, y, x_mask, y_mask)


def make_case(seed=0, n_time=400, n_sensors=17, n_features=3, n_metadata=5, n_samples=37):
    rng = np.random.RandomState(seed)
    return dict(
        data=rng.rand(n_time, n_sensors, n_features) * 60.0,
        metadata=rng.rand(n_sensors, n_metadata),
        input_mask=rng.rand(n_time, n_sensors) > 0.4,
        output_mask=rng.rand(n_time, n_sensors) > 0.3,
        available=(rng.rand(n_sensors, 1) > 0.25).astype(np.float64),
        idx=rng.choice(np.arange(SEQ_LEN - 1, n_time - HORIZON), size=n_samples, replace=False),
    )


def collect(loader):
    return [tuple(np.asarray(a) for a in batch) for batch in loader.get_iterator()]


def assert_same(expected, actual, label=''):
    assert len(expected) == len(actual), f'{label}: batch count {len(expected)} != {len(actual)}'
    names = ('x', 'y', 'x_mask', 'y_mask')
    for b, (exp_batch, act_batch) in enumerate(zip(expected, actual)):
        for name, exp, act in zip(names, exp_batch, act_batch):
            assert exp.shape == act.shape, f'{label} batch {b} {name}: {exp.shape} != {act.shape}'
            assert np.allclose(exp, act, equal_nan=True, rtol=1e-6, atol=1e-6), \
                f'{label} batch {b} {name} differs'


def test_matches_reference():
    """Full option matrix against the old implementation."""
    case = make_case()
    bs = 8  # 37 samples, so the tail sample is dropped exactly as before

    for use_metadata in (False, True):
        for use_masks in (False, True):
            for availability in ('none', 'scale', 'drop'):
                for prefetch_depth in (0, 2):
                    kwargs = dict(
                        metadata=case['metadata'] if use_metadata else None,
                        input_mask=case['input_mask'] if use_masks else None,
                        output_mask=case['output_mask'] if use_masks else None,
                        available_sensors=None if availability == 'none' else case['available'],
                        drop_unavailable_sensors=availability == 'drop',
                    )
                    label = f'metadata={use_metadata} masks={use_masks} ' \
                            f'avail={availability} prefetch={prefetch_depth}'

                    expected = collect(ReferenceDataLoader(
                        case['data'], case['idx'], SEQ_LEN, HORIZON, bs, **kwargs))
                    actual = collect(DataLoader(
                        case['data'], case['idx'], SEQ_LEN, HORIZON, bs, LOGGER,
                        prefetch_depth=prefetch_depth, **kwargs))
                    assert_same(expected, actual, label)


def test_batch_count_and_shapes():
    case = make_case(n_samples=37)
    loader = DataLoader(case['data'], case['idx'], SEQ_LEN, HORIZON, 8, LOGGER,
                        metadata=case['metadata'])
    batches = collect(loader)

    assert loader.num_batch == 4, loader.num_batch  # 37 // 8, the remainder is dropped
    assert len(batches) == 4
    x, y, x_mask, y_mask = batches[0]
    assert x.shape == (8, SEQ_LEN, 17, 3 + 5), x.shape
    assert y.shape == (8, HORIZON, 17, 1), y.shape
    assert x_mask.shape == (8, SEQ_LEN, 17, 1), x_mask.shape
    assert y_mask.shape == (8, HORIZON, 17, 1), y_mask.shape


def test_shuffle_changes_order_but_not_content():
    case = make_case()
    loader = DataLoader(case['data'], case['idx'], SEQ_LEN, HORIZON, 8, LOGGER)
    reference = ReferenceDataLoader(case['data'], case['idx'], SEQ_LEN, HORIZON, 8)

    np.random.seed(7)
    loader.shuffle()
    assert not np.array_equal(loader.idx, case['idx'])
    reference.idx = loader.idx

    assert_same(collect(reference), collect(loader), 'after shuffle')


def test_rejects_out_of_range_windows():
    case = make_case()
    bad_idx = np.append(case['idx'], case['data'].shape[0] - 1)
    try:
        DataLoader(case['data'], bad_idx, SEQ_LEN, HORIZON, 8, LOGGER)
    except AssertionError:
        return
    raise AssertionError('expected an out-of-range idx to be rejected')


def test_abandoned_iterator_does_not_leak_producer_thread():
    case = make_case()
    loader = DataLoader(case['data'], case['idx'], SEQ_LEN, HORIZON, 8, LOGGER, prefetch_depth=2)

    before = threading.active_count()
    iterator = loader.get_iterator()
    next(iterator)
    iterator.close()

    assert threading.active_count() == before, threading.enumerate()
    assert not any(t.name == 'dataloader-prefetch' for t in threading.enumerate())


def test_buffers_are_not_aliased_across_batches():
    """The ring buffers are reused, so consumers must never see two batches share storage."""
    case = make_case()
    loader = DataLoader(case['data'], case['idx'], SEQ_LEN, HORIZON, 8, LOGGER, prefetch_depth=2)
    batches = list(loader.get_iterator())

    assert len(batches) >= 2
    for i in range(len(batches) - 1):
        for a, b in zip(batches[i], batches[i + 1]):
            assert a.data_ptr() != b.data_ptr()


def test_cuda_batches_match_reference():
    if not torch.cuda.is_available():
        print('  (skipped, no CUDA)')
        return

    case = make_case()
    kwargs = dict(metadata=case['metadata'], input_mask=case['input_mask'],
                  output_mask=case['output_mask'], available_sensors=case['available'])
    expected = collect(ReferenceDataLoader(case['data'], case['idx'], SEQ_LEN, HORIZON, 8, **kwargs))

    for prefetch_depth in (0, 2):
        loader = DataLoader(case['data'], case['idx'], SEQ_LEN, HORIZON, 8, LOGGER,
                            device='cuda', prefetch_depth=prefetch_depth, **kwargs)
        actual = []
        for batch in loader.get_iterator():
            assert all(t.is_cuda for t in batch)
            actual.append(tuple(t.cpu().numpy() for t in batch))
        assert_same(expected, actual, f'cuda prefetch={prefetch_depth}')


def main():
    print("Testing dataloader equivalence...\n")

    for test in (
        test_matches_reference,
        test_batch_count_and_shapes,
        test_shuffle_changes_order_but_not_content,
        test_rejects_out_of_range_windows,
        test_abandoned_iterator_does_not_leak_producer_thread,
        test_buffers_are_not_aliased_across_batches,
        test_cuda_batches_match_reference,
    ):
        print(f"  {test.__name__}")
        test()

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
