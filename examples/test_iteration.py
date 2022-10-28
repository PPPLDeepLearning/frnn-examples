#-*- coding: utf-8 -*-

# First attempt at training FRNN-style
import logging
import shutil
import errno
import tempfile
from os import environ

import torch
from torch.utils.data import DataLoader

import sys
sys.path.append("/home/rkube/repos/frnn-loader")

from frnn_loader.primitives.resamplers import resampler_causal
from frnn_loader.backends.backend_hdf5 import backend_hdf5
from frnn_loader.backends.backend_dummy import backend_dummy
from frnn_loader.backends.fetchers import fetcher_d3d_v1
from frnn_loader.primitives.signal import signal_0d
from frnn_loader.loaders.frnn_dataset_disk import shot_dataset_disk
from frnn_loader.loaders.frnn_multi_dataset import frnn_multi_dataset
from frnn_loader.loaders.frnn_loader import batched_random_sequence_sampler

# This specifies the root path under which all HDF5 data will be stored
root = "/projects/FRNN/frnn_loader"
# Work on 2 shots 
shotlist = [180619, 180620]

# Instantiate resampler, etc.
my_resampler = resampler_causal(0.0, 2e3, 1e0)
# Instantiate a file backend. This object handles file storage
my_backend_file = backend_hdf5(root)
# Fetchers handle data fetching from remote targets
my_fetcher = fetcher_d3d_v1()
# Specify a list of predictors. See frnn_loader/frnn_loader/data/d3d_signals.yaml for definitions
pred_list = [signal_0d(n) for n in ["dssdenest", "q95", "echpwrc", "pradcore", "ipsiptargt"]a]

# Generate a list of datasets, one for each shot.
# These datasets cache all data in a hdf5-file on disk.
ds_list = [shot_dataset_disk(shotnr,
                             predictors=pred_list,
                             resampler=my_resampler,
                             backend_file=my_backend_file,
                             fetcher=my_fetcher,
                             root=root,
                             download=True,
                             dtype=torch.float32) for shotnr in shotlist]
# Generate a multi-shot dataset from the list of data-sets(similar to shotlist)
ds_multi = frnn_multi_dataset(ds_list)


def my_collate_fn(input):
    """Reshape list of inputs to torch.tensor

    Input is a list of length 1. The only element is the list returned
    by frnn_multi_dataset.__getitem__

    We
    """
    X_vals = torch.stack([t[0] for t in input[0]])
    Y_vals = torch.stack([t[1] for t in input[0]])

    return X_vals, Y_vals

# This sampler pulls a batch of `batch_size` sequences, each of length `seq_length
# from the dataset. The sequences start at random places and are uniformly distributed
# over the shots of the dataset.
my_sampler = batched_random_sequence_sampler(ds_list, seq_length=100, batch_size=4)
# Default pytorch dataloader. Batch shape is defined through my_sampler
# my_collate_fn is used to transform output from dataset.__getitem__ to a torch.tensor
my_loader = DataLoader(ds_multi, sampler=my_sampler, collate_fn=my_collate_fn)

# Iterate over training set
num_batches = 0
for v in my_loader:
    print(v[0].shape, v[1].shape)
    num_batches += 1
print(f"num_batches = {num_batches}")
