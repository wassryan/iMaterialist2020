
import os
from os import path as osp

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image
from src.rle import kaggle_rle_encode
from src.config import get_config,print_usage
from src.FashionDataset import FashionDataset
from src.metrics import calculateMetrics
import src.transform as T
from models.segmentation import get_model_instance_segmentation
# from MultiEpochDataLoader import MultiEpochsDataLoader, CudaDataLoader

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)

config, _ = get_config()
n_classes = 47
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_test = get_model_instance_segmentation(n_classes)

load_path = osp.join(config.save_dir, 'weights')

ckpt_state = torch.load(osp.join(load_path, '{}_model.bin'.format(config.checkpoint)), map_location='cpu')
pretrain_params = ckpt_state['state_dict']

for k in list(pretrain_params.keys()):
    if k.startswith('module.'):
        pretrain_params[k[len('module.'):]] = pretrain_params[k]
        del pretrain_params[k]

model_test.load_state_dict(pretrain_params)

for param in model_test.parameters():
    param.requires_grad = False

model_test.to(device)
model_test.eval()

config.ann_path = "/".join(config.ann_path.split('/')[:-1]) + "/trainval/validation.csv"
vl_dt = FashionDataset(config, transforms=get_transform(train=False))

test_batch_size = 1
print("Test batch size: ", test_batch_size)
vl_data_loader = DataLoader(
    vl_dt, test_batch_size, shuffle=False,
    num_workers = 24, pin_memory=True, collate_fn=lambda x: tuple(zip(*x))
)

m_iou, m_f1 = calculateMetrics(vl_data_loader, model_test, test_batch_size)

print("mIoU: {:.3f}, mF1: {:.3f}".format(m_iou, m_f1))