import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn
# from coco_utils import get_coco_api_from_dataset
# from coco_eval import CocoEvaluator
import src.utils as utils
from src.metrics import calculateMetrics

def test_on_epoch(vl_data_loader, model, test_batch_size, epoch, device):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    m_iou, m_f1 = calculateMetrics(vl_data_loader, model, test_batch_size, device)
    print('After {epoch} epoch(s), the mean of iou in test set is {m_iou}, the mean of f1 score is {m_f1}'.format(
        epoch=epoch, m_iou=m_iou, m_f1=m_f1))

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq,
    attr=False, 
    loss_weight={
        'loss_classifier':1,
        'loss_box_reg':1,
        'loss_mask':1,
        'loss_objectness':1,
        'loss_rpn_box_reg':1}):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        # losses = sum(loss for loss in loss_dict.values())
        losses = sum(loss * loss_weight[k] for k, loss in loss_dict.items())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss * loss_weight[k] for k, loss in loss_dict_reduced.items())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # ['loss_box_reg', 'loss_classifier', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg', 'loss', 'lr']
        # print(metric_logger.meters['loss_box_reg'].avg, metric_logger.meters['lr'].avg)
        # print(metric_logger.meters['loss_attribute'].avg)
    
    res_metric = {
        'loss_box_reg': metric_logger.meters['loss_box_reg'].avg,
        'loss_classifier': metric_logger.meters['loss_classifier'].avg,
        'loss_mask': metric_logger.meters['loss_mask'].avg,
        'loss_objectness': metric_logger.meters['loss_objectness'].avg,
        'loss': metric_logger.meters['loss'].avg,
        'lr': metric_logger.meters['lr'].avg,
    }
    if attr:
        res_metric['loss_attribute'] = metric_logger.meters['loss_attribute'].avg
    return res_metric


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


# @torch.no_grad()
# def evaluate(model, data_loader, device):
#     n_threads = torch.get_num_threads()
#     # FIXME remove this and make paste_masks_in_image run on the GPU
#     torch.set_num_threads(1)
#     # cpu_device = torch.device("cpu")
#     model.eval()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'
#
#     coco = get_coco_api_from_dataset(data_loader.dataset)
#     iou_types = _get_iou_types(model)
#     coco_evaluator = CocoEvaluator(coco, iou_types)
#
#     for image, targets in metric_logger.log_every(data_loader, 100, header):
#         image = list(img.to(device) for img in image)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#
#         torch.cuda.synchronize()
#         model_time = time.time()
#         outputs = model(image)
#
#         outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
#         model_time = time.time() - model_time
#
#         res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
#         evaluator_time = time.time()
#         coco_evaluator.update(res)
#         evaluator_time = time.time() - evaluator_time
#         metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
#
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     coco_evaluator.synchronize_between_processes()
#
#     # accumulate predictions from all images
#     coco_evaluator.accumulate()
#     coco_evaluator.summarize()
#     torch.set_num_threads(n_threads)
#     return coco_evaluator