#!/usr/bin/env python

import numpy as np
import logging

import torch
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# import torch.nn as nn
# import torch.optim as optim

from tqdm import tqdm

import sys
sys.path.append("/home/rkube/repos/frnn_loader")

import matplotlib.pyplot as plt

logging.basicConfig(filename="test_TTELM.log",
                    format="%(asctime)s    %(message)s",
                    encoding="utf-8",
                    level=logging.INFO)


from frnn_loader.backends.fetchers import fetcher_d3d_v1, fetcher_dummy
from frnn_loader.backends.backend_hdf5 import backend_hdf5
from frnn_loader.primitives.filters import filter_ip_thresh
from frnn_loader.primitives.resamplers import resampler_causal
from frnn_loader.primitives.signal import signal_0d
from frnn_loader.primitives.targets import target_TTELM, target_NULL
from frnn_loader.loaders.frnn_dataset_disk import shot_dataset_disk
from frnn_loader.primitives.normalizers import mean_std_normalizer

from frnn_loader.utils.errors import SignalCorruptedError, NotDownloadedError



root = "/home/rkube/datasets/frnn"
# 1/ Describe the dataset
predictor_tags = (
    "q95",
    "efsli",
    "ipspr15V",
    "fs07",
    "efsbetan",
    "efswmhd",
    "dssdenest",
    "pradcore",
    "pradedge",
    "bmspinj",
    "bmstinj",
    "ipsiptargt",
    "ipeecoil",
)
predictor_list = tuple([signal_0d(tag) for tag in predictor_tags])


shotnr = 174829


# Instantiate the filter we use to crimp the shot times
dt = 1.0 # Time used for resampling
ip_filter = filter_ip_thresh(0.2)
signal_ip = signal_0d("ipspr15V")
my_backend = backend_hdf5(root)
my_fetcher = fetcher_dummy() #fetcher_d3d_v1()
my_resampler = resampler_causal(500.0, 5000.0, dt)

ds = shot_dataset_disk(shotnr, 
    predictors=predictor_list,
    resampler=my_resampler,
    backend_file=my_backend,
    fetcher=my_fetcher,
    root=root,
    download=True,
    normalizer=None,
    is_disruptive=False,
    target=target_NULL,
    dtype=torch.float32)


my_normalizer = mean_std_normalizer()
my_normalizer.fit([ds])
print(my_normalizer)

ds_norm = shot_dataset_disk(shotnr, 
    predictors=predictor_list,
    resampler=my_resampler,
    backend_file=my_backend,
    fetcher=my_fetcher,
    root=root,
    download=True,
    normalizer=my_normalizer,
    is_disruptive=False,
    target=target_TTELM,
    dtype=torch.float32)

data, target = ds[:]
data_n, target_n = ds_norm[:]

plt.figure()
plt.plot(target_n)
plt.plot(data_n[:,3])
plt.show()

#ds_norm.delete_data_file()
#ds.delete_data_file()



