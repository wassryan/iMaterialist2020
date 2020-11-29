import os
from os import path as osp

import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from src.engine import train_one_epoch, test_on_epoch
from src.FashionDataset import FashionDataset
from src.config import get_config, print_usage
import src.transform as T
from src.metrics import calculateMetrics
from models.segmentation import get_model_instance_segmentation
from PIL import Image,ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if torch.cuda.device_count() > 1:
	print("Use Device Number: {}".format(torch.cuda.device_count()))

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)

def train(config):
    print(config)
    ### tensorboard ###
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    tbwriter = SummaryWriter(config.log_dir)

    # create our own dataset and its data_loader
    tr_dt = FashionDataset(config, get_transform(train = True))
    # config.ann_path = config.ann_path[:-9]+'validation.csv'
    config.ann_path = "/".join(config.ann_path.split('/')[:-1]) + "/trainval/validation.csv"
    print("***" * 40)
    print(config.ann_path)

    vl_dt = FashionDataset(config, get_transform(train=False))
    config.ann_path = "/".join(config.ann_path.split('/')[:-1]) + "/trainval/train.csv"
    print("***" * 40)
    print(config.ann_path)

    # need pin_memory=True to allow a bigger batch size
    tr_data_loader = DataLoader(
        tr_dt, config.batch_size , shuffle = True,
        num_workers = 32, pin_memory=True, collate_fn = lambda x: tuple(zip(*x))
    )

    test_batch_size = config.batch_size // torch.cuda.device_count()
    print("Test batch size: ", test_batch_size)
    vl_data_loader = DataLoader(
        vl_dt, test_batch_size, shuffle=False,
        num_workers = 32, pin_memory=True, collate_fn=lambda x: tuple(zip(*x))
    )

    # save the weights
    weight_dir = osp.join(config.save_dir, 'weights')
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    # there are 46 classes in total
    num_classes = 46 + 1
    # create model instance
    model = get_model_instance_segmentation(num_classes)
    print(model)

    #set model to device
    model.to(device)
    model = DataParallel(model)

    # for optim
    params = [p for p in model.parameters() if p.requires_grad]

    optim = torch.optim.SGD(params, lr = config.lr, momentum=0.9, weight_decay=0.0005)

    if config.scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.1)
    elif config.scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=(config.num_epochs // 5) + 1, eta_min=1e-06)

    for epoch in range(config.num_epochs):
        res_metric = train_one_epoch(model, optim, tr_data_loader, device, epoch, print_freq=config.rep_intv)
        tbwriter.add_scalar('train/loss', res_metric['loss'], epoch)
        tbwriter.add_scalar('train/loss_box_reg', res_metric['loss_box_reg'], epoch)
        tbwriter.add_scalar('train/loss_classifier', res_metric['loss_classifier'], epoch)
        tbwriter.add_scalar('train/loss_mask', res_metric['loss_mask'], epoch)
        tbwriter.add_scalar('train/loss_objectness', res_metric['loss_objectness'], epoch)
        tbwriter.add_scalar('train/lr', res_metric['lr'], epoch)
        # updt the learning rate
        lr_scheduler.step()
        w1 = osp.join(config.save_dir , 'weights')
        wfile = osp.join(w1, '{}_model.bin'.format(str(epoch)))

        # test_on_epoch(vl_data_loader, model, test_batch_size, epoch, device)

        # torch.save(model.state_dict(), wfile)
        torch.save({
            'state_dict': model.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'optim': optim.state_dict(),
            'epoch': epoch,
        }, wfile)
        print('=> Save {}...'.format(wfile))


if __name__ == '__main__':
    # parse configuration
    config, unparsed = get_config()
    if len(unparsed)>0:
        print_usage()
        exit(1)

    train(config)