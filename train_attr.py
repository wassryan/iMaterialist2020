import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from os import path as osp
import os
from src.engine import train_one_epoch
from models.segmentation_fine_grained import get_model
from src.FashionDataset import FashionDatasetwithAttr
from src.config import get_config, print_usage
import src.transform as T
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
    tr_dt = FashionDatasetwithAttr(config, get_transform(train = True))
    # need pin_memory=True to allow a bigger batch size
    tr_data_loader = DataLoader(
        tr_dt, config.batch_size , shuffle = True,
        num_workers = 32, pin_memory=True, collate_fn = lambda x: tuple(zip(*x))
    )

    # save the weights
    weight_dir = osp.join(config.save_dir, 'weights')

    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    # there are 46 classes in total
    num_classes = 46 + 1
    # create model instance
    model = get_model(nr_class=num_classes, attr_score_thresh=config.attr_score_thresh, pos_weight=config.pos_weight)
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
        loss_weight = {
                'loss_classifier':1,
                'loss_box_reg':1,
                'loss_mask':1,
                'loss_objectness':1,
                'loss_rpn_box_reg':1
            }
        attr_flag = True
        if attr_flag:
            loss_weight['loss_attribute'] = config.loss_aweight
        res_metric = train_one_epoch(model, optim, tr_data_loader, device, epoch, print_freq=config.rep_intv,
            attr=attr_flag, loss_weight=loss_weight)
        tbwriter.add_scalar('train/loss', res_metric['loss'], epoch)
        tbwriter.add_scalar('train/loss_box_reg', res_metric['loss_box_reg'], epoch)
        tbwriter.add_scalar('train/loss_classifier', res_metric['loss_classifier'], epoch)
        tbwriter.add_scalar('train/loss_mask', res_metric['loss_mask'], epoch)
        tbwriter.add_scalar('train/loss_objectness', res_metric['loss_objectness'], epoch)
        tbwriter.add_scalar('train/loss_attribute', res_metric['loss_attribute'], epoch)
        tbwriter.add_scalar('train/lr', res_metric['lr'], epoch)
        # updt the learning rate
        lr_scheduler.step()
        w1 = osp.join(config.save_dir , 'weights')
        wfile = osp.join(w1, '{}_model.bin'.format(str(epoch)))

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