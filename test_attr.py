import torch
from os import path as osp
import os
import numpy as np
import pandas as pd
from models.segmentation_fine_grained import get_model
from src.config import get_config,print_usage
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image
from src.rle import kaggle_rle_encode, rle_encode,rle_to_string
from tqdm import tqdm
import cv2
import math
import pickle

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load_pkl(pkl_name):
    with open(pkl_name + '.pkl', 'rb') as f:
        return pickle.load(f)

def _scale_image(img, long_size):
    if img.shape[0] < img.shape[1]:
        scale = img.shape[1] / long_size
        size = (long_size, math.floor(img.shape[0] / scale))
    else:
        scale = img.shape[0] / long_size
        size = (math.floor(img.shape[1] / scale), long_size)
    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

def refine_masks(masks,labels, im):
    # compute the areas of each mask
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis = 0)
    # ordered masks from smallest to largest
    mask_index = np.argsort(areas)
    # one reference mask is created to be incrementally populated
    union_mask = {k: np.zeros(masks.shape[:-1], dtype = bool) for k in np.unique(labels)}

    for m in mask_index:
        label = labels[m]
        masks[:,:,m] = np.logical_and(masks[:,:, m], np.logical_not(union_mask[label]))
        union_mask[label] = np.logical_or(masks[:,:,m], union_mask[label])

    # reorder masks
    refined = list()

    for m in range(masks.shape[-1]):
        mask_raw = cv2.resize(masks[:,:,m], (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        #mask = mask_raw.ravel(order='F')
        rle = kaggle_rle_encode(mask_raw)
        #rle = rle_encode(mask_raw)
        label = labels[m] - 1
        refined.append([mask_raw, rle, label])

    return refined

def refine_masksv2(masks, labels, im, attrs, attrmap_dict):
    # compute the areas of each mask
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis = 0)
    # ordered masks from smallest to largest
    mask_index = np.argsort(areas)
    # one reference mask is created to be incrementally populated
    union_mask = {k: np.zeros(masks.shape[:-1], dtype = bool) for k in np.unique(labels)}

    for m in mask_index:
        label = labels[m]
        masks[:,:,m] = np.logical_and(masks[:,:, m], np.logical_not(union_mask[label]))
        union_mask[label] = np.logical_or(masks[:,:,m], union_mask[label])

    # reorder masks
    refined = list()

    for m in range(masks.shape[-1]): # proposal number
        mask_raw = cv2.resize(masks[:,:,m], (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        #mask = mask_raw.ravel(order='F')
        rle = kaggle_rle_encode(mask_raw)
        #rle = rle_encode(mask_raw)
        label = labels[m] - 1
        ### deal with attribute id ###
        attr = attrs[m]
        attr = attr.cpu().numpy().tolist()

        ### preprocess prediction attribute id ###
        # remove 0 labels if len(attr) > 1
        # set [] if len(attr) == 1 && attr[0]==0
        if (0 in attr) and len(attr) > 1:
            attr.remove(0)
        elif (len(attr) == 1) and (attr[0] == 0):
            attr = []

        ### translate to string ###
        attr_str = ','.join(str(attrmap_dict[x]) for x in attr)
        # set_trace()
        refined.append([mask_raw, rle, label, attr_str])

    return refined

def test(config):
    print(config)
    # test_dt = FashionDataset(config, transforms= None)
    sample_df = pd.read_csv(config.sample_path)

    ### Map Table ###
    root='/data/imaterialist2020'
    map_file = root + '/attr_map'
    attrmap_dict = load_pkl(map_file)
    ################################################################################
    # create the model instance
    num_classes = 46 + 1
    # model_test = get_model_instance_segmentation(num_classes)
    model_test = get_model(nr_class=num_classes, attr_score_thresh=config.attr_score_thresh)

    #load the training weights
    # load_path =osp.join(config.save_dir, '9_weights')
    load_path = osp.join(config.save_dir, 'weights')
    # pretrain_params = torch.load(osp.join(load_path, '{}_model.bin'.format(config.checkpoint)),map_location='cpu')
    ckpt_state = torch.load(osp.join(load_path, '{}_model.bin'.format(config.checkpoint)), map_location='cpu')
    pretrain_params = ckpt_state['state_dict']
    # print(pretrain_params)
    for k in list(pretrain_params.keys()):
        if k.startswith('module.'):
            pretrain_params[k[len('module.'):]] = pretrain_params[k]
            del pretrain_params[k]

    model_test.load_state_dict(pretrain_params)

    # send the test model to gpu
    model_test.to(device)

    for param in model_test.parameters():
        param.requires_grad = False

    model_test.eval()

    # for submission
    sub_list = []
    missing_count = 0

    for i,row in tqdm(sample_df.iterrows(), total = len(sample_df)):
        ###modify##########################################################
        # import the image
        img_path = osp.join(config.test_dir,sample_df['ImageId'][i]+'.jpg')
        # print(img_path)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((config.width,config.height), resample = Image.BILINEAR)
        #  convert the img as tensor
        img = F.to_tensor(img)
        #####modify#############################################################
        # labels/scores/boxes: box branch, masks: mask branch
        pred = model_test([img.to(device)])[0] # {'labels:', 'masks', 'scores', 'boxes'}
        masks = np.zeros((512,512, len(pred['masks'])))

        for j,m in enumerate(pred['masks']):
            res = transforms.ToPILImage()(m.permute(1,2,0).cpu().numpy())
            res = np.asarray(res.resize((512, 512), resample=Image.BILINEAR))

            masks[:,:,j] = (res[:,:] * 255. > 127).astype(np.uint8)

        labels = pred['labels'].cpu().numpy() # (nr_proposals,)
        scores = pred['scores'].cpu().numpy()
        ascores = pred['ascores'] # .cpu().numpy()
        # print("ascores: ", ascores)
        # print("scores: ", scores)
        # print("labels: ", labels)
        # set_trace()
        best_idx = 0
        # print('the maximum scores is {}'.format(np.mean(scores)))
        # print('the current masks is {}'.format(masks))
        for _scores in scores:
            if _scores > config.mask_thresh:
                best_idx += 1

        if best_idx == 0:
            # print(masks.shape[-1])
            sub_list.append([sample_df.loc[i, 'ImageId'], '1 1', 23, ''])
            missing_count += 1
            continue
        # mask在roi_heads部分做了后处理，将maskpaste到原图输入上: masks(512,512,nr_proposals)
        # 根据box branch预测得到的labels取对应mask channel的mask作为最终的mask
        if masks.shape[-1]>0:
            im = cv2.imread(img_path)
            im = _scale_image(im, 1024)
            # FIXME: refine_masks ????
            # masks = refine_masks(masks[:,:,:best_idx], labels[:best_idx], im) # List([mask_raw, rle, label])
            masks = refine_masksv2(masks[:,:,:best_idx], labels[:best_idx], im, ascores[:best_idx], attrmap_dict['new2attr'])
            for m, rle, label, attr in masks:
                # print(attr)
                # set_trace()
                # sub_list.append([sample_df.loc[i, 'ImageId'], rle, label, '']) # TODO: attribute assign
                sub_list.append([sample_df.loc[i, 'ImageId'], rle, label, attr])
        else:
            sub_list.append([sample_df.loc[i, 'ImageId'], '1 1', 23, '']) # TODO: attribute assign
            missing_count += 1
        #if i > 2:
        #    break
    #set_trace()
    submission_df = pd.DataFrame(sub_list, columns=sample_df.columns.values)
    print("Total image results: ", submission_df['ImageId'].nunique())
    print("Missing Images: ", missing_count)
    submission_df = submission_df[submission_df.EncodedPixels.notnull()]
    # for row in range(len(submission_df)):
    #     line = submission_df.iloc[row, :]
    #     submission_df.iloc[row, 1] = line['EncodedPixels'].replace('.0', '')
    # # submission_df.head()
    submit_path = config.submit_path + 'submission_{}e_{}t_{}attr.csv'.format(
        config.checkpoint, config.mask_thresh, config.attr_score_thresh)
    print('submit_path: ',submit_path)
    submission_df.to_csv(submit_path, index=False)
    print('ok,finished')


if __name__ == '__main__':
    # parse configuration
    config, unparsed = get_config()
    #print(config)
    if len(unparsed)>0:
        print_usage()
        exit(1)

    test(config)
