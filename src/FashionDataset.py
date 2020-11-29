import os
import torch
from os import path as osp
import numpy as np
import pandas as pd
import collections
from PIL import Image
from tqdm import tqdm
from src.config import *
from src.rle import kaggle_rle_decode
from pycocotools import mask as mutils
import math
import pickle

# slightly modifications on https://www.kaggle.com/abhishek/mask-rcnn-using-torchvision-0-17
class FashionDataset(torch.utils.data.Dataset):
    def __init__(self,config,transforms = None):

        self.image_dir = config.img_dir
        self.annotation = pd.read_csv(config.ann_path)

        # resize the img of width and height
        self.width = config.width #512
        self.height = config.height #512
        self.transforms = transforms
        self.train_dict = collections.defaultdict(dict)

        #create a temp_dataframe for extraction of useful data
        self.annotation['CategoryId'] = self.annotation.ClassId.apply(lambda x: str(x).split('_')[0])
        #[[region1],[region2]],[classid1,classid2]
        df = self.annotation.groupby('ImageId')['EncodedPixels','CategoryId'].agg(lambda x: list(x)).reset_index()
        size = self.annotation.groupby('ImageId')['Height','Width'].mean().reset_index()
        df = df.merge(size, on = 'ImageId', how ='left')
         
        for idx, row in tqdm(df.iterrows(), total = len(df)):
            self.train_dict[idx]['image_id'] = row['ImageId']
            self.train_dict[idx]['image_path'] = osp.join(self.image_dir, row['ImageId']) + '.jpg'
            self.train_dict[idx]['labels'] = row['CategoryId']
            self.train_dict[idx]['height'] = self.height
            self.train_dict[idx]['width'] = self.width
            self.train_dict[idx]['orig_height'] = row['Height']
            self.train_dict[idx]['orig_width'] = row['Width']
            self.train_dict[idx]['annotations'] = row['EncodedPixels']

    def __getitem__(self, idx):
        # load images as masks
        img_path = self.train_dict[idx]['image_path']
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.width,self.height),resample = Image.BILINEAR)

        train_data = self.train_dict[idx]
        # for gpu, it is better for np.uint8
        mask = np.zeros((len(train_data['annotations']),self.width,self.height), dtype = np.uint8) # (nr_region, w, h)

        labels = []

        for ind, (ann, label) in enumerate(zip(train_data['annotations'], train_data['labels'])):

            sub_mask = kaggle_rle_decode(ann, train_data['orig_height'], train_data['orig_width'])
            # to convert array to image
            sub_mask = Image.fromarray(sub_mask)
            # resize the image to (512,512)
            sub_mask = sub_mask.resize((self.width, self.height),resample=Image.BILINEAR)
            mask[ind,:,:] = sub_mask
            # 0 is for background
            labels.append(int(label)+1)

        # get bounding box coordinates for each mask
        num_objs = len(labels)
        boxes = []
        #### make a reference to https://www.kaggle.com/abhishek/mask-rcnn-using-torchvision-0-17 #####
        new_labels = []
        new_masks = []

        for i in range(num_objs):
            try:
                pos = np.where(mask[i, :, :]) # tuple(row_idx, col_idx)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if abs(xmax - xmin) >= 20 and abs(ymax - ymin) >= 20:
                    boxes.append([xmin, ymin, xmax, ymax])
                    new_labels.append(labels[i])
                    new_masks.append(mask[i, :, :])
            except ValueError:
                continue

        if len(new_labels) == 0:
            boxes.append([0, 0, 20, 20])
            new_labels.append(0)
            new_masks.append(mask[0, :, :])
        ######################################################################################################
        # get the final masks
        # TODO: duplicate operation?
        final_masks = np.zeros((len(new_masks),self.width, self.height), dtype = np.uint8)

        for ind, _m in enumerate(new_masks):
            final_masks[ind,:,:] = _m

        # convert everything into a tensor.Tensor
        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        labels = torch.as_tensor(new_labels, dtype = torch.int64)
        masks = torch.as_tensor(final_masks, dtype = torch.uint8) # (nr_objs, 512, 512)

        image_id = torch.tensor([idx])

        # calculate bounding box areas
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2]- boxes[:,0])
        # to check the instance is crowded or not (assume it is single instance)
        iscrowd = torch.zeros((num_objs, ), dtype = torch.int64)


        target = {}
        target['boxes'] = boxes # (nr_objs,4)
        target['labels'] = labels # (nr_objs,)
        target['masks'] = masks # (nr_objs,512,512)
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        # print(boxes.shape, labels.shape, masks.shape)

        if self.transforms is not None:
            img, target = self.transforms(img, target)


        return img, target

    def __len__(self):
        return len(self.train_dict)

class FashionDatasetwithAttr(torch.utils.data.Dataset):
    def __init__(self,config,transforms = None):

        self.image_dir = config.img_dir
        # self.test_dir = config.test_dir
        self.annotation = pd.read_csv(config.ann_path)

        def load_pkl(pkl_name):
            with open(pkl_name + '.pkl', 'rb') as f:
                return pickle.load(f)
        self.attr_map = load_pkl(config.submit_path + 'attr_map')

        # resize the img of width and height
        self.width = config.width #512
        self.height = config.height #512
        self.transforms = transforms
        self.train_dict = collections.defaultdict(dict)

        def str2int(x):
            return int(x)

        def splitAttr(x):
            if isinstance(x, str):
                return list(map(str2int, str(x).split(',')))
            if math.isnan(x): # abnormal value(no attribute for the object)
                return [-1]
            else:
                print("bad input")

        #create a temp_dataframe for extraction of useful data
        self.annotation['CategoryId'] = self.annotation.ClassId.apply(lambda x: str(x).split('_')[0])
        self.annotation['AttributesIds'] = self.annotation.AttributesIds.apply(lambda x: splitAttr(x))
        #[[region1],[region2]],[classid1,classid2]
        df = self.annotation.groupby('ImageId')['EncodedPixels','CategoryId', 'AttributesIds'].agg(lambda x: list(x)).reset_index()
        size = self.annotation.groupby('ImageId')['Height','Width'].mean().reset_index()
        df = df.merge(size, on = 'ImageId', how ='left')
         
        for idx, row in tqdm(df.iterrows(), total = len(df)):
            self.train_dict[idx]['image_id'] = row['ImageId']
            self.train_dict[idx]['image_path'] = osp.join(self.image_dir, row['ImageId']) + '.jpg'
            self.train_dict[idx]['labels'] = row['CategoryId'] # list(int, int, ...)
            self.train_dict[idx]['attrs'] = row['AttributesIds'] # list(list(int,int,..))
            self.train_dict[idx]['height'] = self.height
            self.train_dict[idx]['width'] = self.width
            self.train_dict[idx]['orig_height'] = row['Height']
            self.train_dict[idx]['orig_width'] = row['Width']
            self.train_dict[idx]['annotations'] = row['EncodedPixels'] # list(str,str, ...) each str represents a region

    def __getitem__(self, idx):
        # load images as masks
        img_path = self.train_dict[idx]['image_path']
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.width,self.height),resample = Image.BILINEAR)

        train_data = self.train_dict[idx]
        # for gpu, it is better for np.uint8
        mask = np.zeros((len(train_data['annotations']),self.width,self.height), dtype = np.uint8) # (nr_region, w, h)

        labels = []
        attrs = []

        for ind, (ann, label, attr) in enumerate(zip(train_data['annotations'], train_data['labels'], train_data['attrs'])):
            # ann: str, label: int, attr: list(int,int, ...)
            sub_mask = kaggle_rle_decode(ann, train_data['orig_height'], train_data['orig_width'])
            # to convert array to image
            sub_mask = Image.fromarray(sub_mask)
            # resize the image to (512,512)
            sub_mask = sub_mask.resize((self.width, self.height),resample=Image.BILINEAR)
            mask[ind,:,:] = sub_mask
            # 0 is for background
            labels.append(int(label)+1)
            attrs.append(list(map(lambda x: self.attr_map['attr2new'][x], attr)))

        # get bounding box coordinates for each mask
        num_objs = len(labels)
        boxes = []
        #### make a reference to https://www.kaggle.com/abhishek/mask-rcnn-using-torchvision-0-17 #####
        new_attrs = []
        new_labels = []
        new_masks = []

        for i in range(num_objs):
            try:
                pos = np.where(mask[i, :, :]) # tuple(row_idx, col_idx)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if abs(xmax - xmin) >= 20 and abs(ymax - ymin) >= 20:
                    boxes.append([xmin, ymin, xmax, ymax])
                    new_labels.append(labels[i])
                    new_attrs.append(attrs[i])
                    new_masks.append(mask[i, :, :])
            except ValueError:
                continue

        if len(new_labels) == 0:
            boxes.append([0, 0, 20, 20])
            new_labels.append(0)
            new_attrs.append([0])
            new_masks.append(mask[0, :, :])
        ######################################################################################################
        # get the final masks
        # TODO: duplicate operation?
        final_masks = np.zeros((len(new_masks),self.width, self.height), dtype = np.uint8)

        for ind, _m in enumerate(new_masks):
            final_masks[ind,:,:] = _m

        # convert everything into a tensor.Tensor
        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        labels = torch.as_tensor(new_labels, dtype = torch.int64)
        masks = torch.as_tensor(final_masks, dtype = torch.uint8) # (nr_objs, 512, 512)
        ### transform into multiple hot vector ###
        # (nr_objs, 294+1)
        attr_tensor = torch.zeros((len(new_attrs), 295), dtype=torch.int64)
        for i, attr in enumerate(new_attrs):
            for idx in attr:
                attr_tensor[i][idx] = 1

        image_id = torch.tensor([idx])

        # calculate bounding box areas
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2]- boxes[:,0])
        # to check the instance is crowded or not (assume it is single instance)
        iscrowd = torch.zeros((num_objs, ) ,dtype = torch.int64)

        target = {}
        target['boxes'] = boxes # (nr_objs,4)
        target['labels'] = labels # (nr_objs,)
        target['masks'] = masks # (nr_objs,512,512)
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        target['attrs'] = attr_tensor
        # print(boxes.shape, labels.shape, masks.shape)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.train_dict)

class FashionDatasetOffline(torch.utils.data.Dataset):
    def __init__(self,config,transforms = None):

        self.image_dir = config.img_dir
        # self.test_dir = config.test_dir
        self.annotation = pd.read_csv(config.ann_path)

        # resize the img of width and height
        self.width = config.width #512
        self.height = config.height #512
        self.transforms = transforms
        self.train_dict = collections.defaultdict(dict)

        #create a temp_dataframe for extraction of useful data
        self.annotation['CategoryId'] = self.annotation.ClassId.apply(lambda x: str(x).split('_')[0])
        #[[region1],[region2]],[classid1,classid2]
        df = self.annotation.groupby('ImageId')['EncodedPixels','CategoryId'].agg(lambda x: list(x)).reset_index()
        size = self.annotation.groupby('ImageId')['Height','Width'].mean().reset_index()
        df = df.merge(size, on = 'ImageId', how ='left')
         
        for idx, row in tqdm(df.iterrows(), total = len(df)):
            self.train_dict[idx]['image_id'] = row['ImageId']
            self.train_dict[idx]['image_path'] = osp.join(self.image_dir, row['ImageId']) + '.jpg'
            self.train_dict[idx]['labels'] = row['CategoryId']
            self.train_dict[idx]['height'] = self.height
            self.train_dict[idx]['width'] = self.width
            self.train_dict[idx]['orig_height'] = row['Height']
            self.train_dict[idx]['orig_width'] = row['Width']
            self.train_dict[idx]['annotations'] = row['EncodedPixels']

    def __getitem__(self, idx):
        # load images as masks
        img_path = self.train_dict[idx]['image_path']
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.width,self.height),resample = Image.BILINEAR)

        train_data = self.train_dict[idx]
        # for gpu, it is better for np.uint8
        mask = np.zeros((len(train_data['annotations']),self.width,self.height), dtype = np.uint8) # (nr_region, w, h)

        labels = []

        for ind, (ann, label) in enumerate(zip(train_data['annotations'], train_data['labels'])):

            sub_mask = kaggle_rle_decode(ann, train_data['orig_height'], train_data['orig_width'])
            # to convert array to image
            sub_mask = Image.fromarray(sub_mask)
            # resize the image to (512,512)
            sub_mask = sub_mask.resize((self.width, self.height),resample=Image.BILINEAR)
            mask[ind,:,:] = sub_mask
            # 0 is for background
            labels.append(int(label)+1)

        # get bounding box coordinates for each mask
        num_objs = len(labels)
        boxes = []
        #### make a reference to https://www.kaggle.com/abhishek/mask-rcnn-using-torchvision-0-17 #####
        new_labels = []
        new_masks = []

        for i in range(num_objs):
            try:
                pos = np.where(mask[i, :, :]) # tuple(row_idx, col_idx)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if abs(xmax - xmin) >= 20 and abs(ymax - ymin) >= 20:
                    boxes.append([xmin, ymin, xmax, ymax])
                    new_labels.append(labels[i])
                    new_masks.append(mask[i, :, :])
            except ValueError:
                continue

        if len(new_labels) == 0:
            boxes.append([0, 0, 20, 20])
            new_labels.append(0)
            new_masks.append(mask[0, :, :])
        ######################################################################################################
        # get the final masks

        final_masks = np.zeros((len(new_masks),self.width, self.height), dtype = np.uint8)

        for ind, _m in enumerate(new_masks):
            final_masks[ind,:,:] = _m

        # convert everything into a tensor.Tensor
        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        labels = torch.as_tensor(new_labels, dtype = torch.int64)
        masks = torch.as_tensor(final_masks, dtype = torch.uint8) # (nr_objs, 512, 512)

        image_id = torch.tensor([idx])

        # calculate bounding box areas
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2]- boxes[:,0])
        # to check the instance is crowded or not (assume it is single instance)
        iscrowd = torch.zeros((num_objs, ) ,dtype = torch.int64)


        target = {}
        target['boxes'] = boxes # (nr_objs,4)
        target['labels'] = labels # (nr_objs,)
        target['masks'] = masks # (nr_objs,512,512)
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        # print(boxes.shape, labels.shape, masks.shape)

        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

        self.train_dict[idx]['target'] = target

        return img, target

    def __len__(self):
        return len(self.train_dict)

if __name__ == '__main__':
    import os
    from config import get_config, print_usage
    # from ipdb import set_trace
    import numpy as np
    config, unparsed = get_config()
    if len(unparsed)>0:
        print_usage()
        exit(1)
    # save_file = os.path.join(config.submit_path, 'train_file.npy') # 6hours
    # print("=> File will be save in :", save_file)

    tr_dt = FashionDatasetwithAttr(config)

    data_len = len(tr_dt)
    print(data_len)
    cnt = 1
    for im, target in tqdm(tr_dt):
        # set_trace()
        # print(target['mask'].shape)
        if cnt > data_len:
            break
        cnt = cnt + 1

    # tr_dt = FashionDatasetOffline(config) # no transform
    
    # # print(len(tr_dt))
    # data_len = len(tr_dt)
    # print(data_len)
    # cnt = 1
    # for im, target in tqdm(tr_dt):
    #     # set_trace()
    #     # print(target['mask'].shape)
    #     if cnt > data_len:
    #         break
    #     cnt = cnt + 1
    # set_trace()
    # np.save(save_file, tr_dt.train_dict)