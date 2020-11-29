# -*- coding: utf-8 -*-
import os
import sys
import colorsys
import random
import math
import pickle
from PIL import Image
import cv2 
import numpy as np
import json

import matplotlib as mpl
mpl.use('PS')
import matplotlib.pyplot as plt

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import *

import torch
from torchvision.transforms import functional as F
from torchvision import transforms
from src.rle import kaggle_rle_encode
from src.draw_mask import draw_masks
from models.segmentation_fine_grained import get_model
plt.axis('off')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Ui_MainWindow(QWidget):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()

        self.map_file = './attr_map'
        self.attrmap_dict = self.load_pkl(self.map_file)
        #print(self.attrmap_dict)

        self.num_classes = 46 + 1
        self.model_test = get_model(nr_class=self.num_classes, attr_score_thresh=0.95)
        self.model_path = './20_model.bin'
        self.ckpt_state = torch.load(self.model_path,map_location='cpu')
        self.pretrain_params = self.ckpt_state['state_dict']
        for k in list(self.pretrain_params.keys()):
            if k.startswith('module.'):
                self.pretrain_params[k[len('module.'):]] = self.pretrain_params[k]
                del self.pretrain_params[k]

        self.model_test.load_state_dict(self.pretrain_params)
        # send the test model to gpu
        self.model_test.to(device)
        for param in self.model_test.parameters():
            param.requires_grad = False
            
        self.model_test.eval()
        #self.config, self.unparsed = get_config()

        self.image_name = None
        self.raw_image = None
        self.logo_size = (350, 350)
        self.mask_size = (60,60)
        self.width = 512
        self.height = 512

        self.pred = None
        self.classes = None
        self.save_path = './images_pred/'

        with open('label_descriptions.json') as json_f:
            look_up_classes = json.load(json_f)    
            self.categories = look_up_classes["categories"]
            self.attributes = look_up_classes["attributes"]

        if not os.path.exists('./images_pred/'):
            os.makedirs('./images_pred/')

        self.setupUi()

    #     Generate random colors.
    #     To get visually distinct colors, generate them in HSV space then
    #     convert to RGB.

    # Return where to text class 
    def text_image(self,prediction,class_id):
        text = np.argwhere(prediction == class_id)
        text_x = text[int(len(text)/2)][1]
        text_y = text[int(len(text)/2)][0]
        return [text_x,text_y ]

    def setupUi(self):
        imgName = './assets/logo.jpg'

        self.setWindowTitle("Fashion is all")
        self.resize(775, 721)
        self.setFixedSize(775, 721)

        font = QtGui.QFont()
        font.setFamily('Arial Black')
        font.setBold(True)
        font.setPointSize(15)
        font.setWeight(20)

        #open image button
        self.pushButton_openImage = QPushButton(QIcon("./assets/open.png"),' Select Image')
        self.pushButton_openImage.clicked.connect(self.openImage)
        self.pushButton_openImage.setObjectName("pushButton_openImage")
        self.pushButton_openImage.setFont(font)
        self.pushButton_openImage.setStyleSheet(" width: 40px;height: 40px;"
                                                "background-color: rgb(200, 200, 200);border:2px groove gray;"
                                                "border-radius: 8px;border-style: solid;border-top-width: 5px; ")

        #logo img
        self.logo_png = QtGui.QPixmap(imgName)
        self.logo_png = self.logo_png.scaled(self.logo_size[1], self.logo_size[0])

        self.mask_set = QtGui.QPixmap(imgName)
        self.mask_set = self.mask_set.scaled(self.mask_size[1], self.mask_size[0])

        #show the original pic
        self.label_image = QLabel(self)
        self.label_image.setPixmap(self.logo_png)
        self.label_image.resize(self.logo_size[1], self.logo_size[0])

        #save image button
        self.pushButton_saveImage = QPushButton(QIcon("./assets/segment.png"),'  Segment')
        self.pushButton_saveImage.clicked.connect(self.saveImage)
        self.pushButton_saveImage.setObjectName("pushButton_saveImage")
        self.pushButton_saveImage.setFont(font)
        self.pushButton_saveImage.setStyleSheet(" width: 40px;height: 40px;"
                                                "background-color: rgb(200, 200, 200);border:2px groove gray;"
                                                "border-radius: 8px;border-style: solid;border-top-width: 5px; ")


        self.exit_button = QPushButton(QIcon("./assets/exit.png"),'  Exit')
        self.exit_button.setFont(font)
        self.exit_button.setStyleSheet(" width: 40px;height: 40px;"
                                                "background-color: rgb(200, 200, 200);border:2px groove gray;"
                                                "border-radius: 8px;border-style: solid;border-top-width: 5px; ")

        self.exit_button.clicked.connect(self.exit_ARC)

        #show the mask image
        self.mask_image = QLabel(self)
        self.mask_image.setPixmap(self.logo_png)
        self.mask_image.resize(self.logo_size[1], self.logo_size[0])

        self.setlayout()
        QtCore.QMetaObject.connectSlotsByName(self)
       
    #exit button    
    def exit_ARC(self):
        QtCore.QCoreApplication.instance().quit() # exit window
        print("=> Happy Ending...")

    #total layout
    def setlayout(self):
        self.vbox = QVBoxLayout()
        #self.vbox.addStretch(1)
        self.vbox_right = QVBoxLayout()
        #self.vbox_right.addStretch(1)
        self.vbox_right.addWidget(self.pushButton_openImage)
        self.vbox_right.addWidget(self.pushButton_saveImage)
        self.vbox_right.addWidget(self.exit_button)
        self.vbox.addLayout(self.vbox_right)
        
        #grid
        self.grid = QGridLayout()
        self.grid.addWidget(self.label_image, 0, 0)   
        self.grid.addWidget(self.mask_image, 1, 0)

        self.grid.addLayout(self.vbox,0,1)
        #self.grid.addWidget(self.mask_txt, 1, 1)

        self.setLayout(self.grid)
        #total = self.grid.count()
        #print('*************' + str(total))
        
    #choose image to upload
    def openImage(self):
        try: 
            global imgName  
            imgName, imgType = QFileDialog.getOpenFileName(self, "open image", "", "*.jpg;;*.png;;All Files(*)")
            self.raw_image = Image.open(imgName)
            # resize image
            #scale_radio = self.logo_size[0]/self.raw_image.size[0]
            #jpg = QtGui.QPixmap(imgName).scaled(self.logo_size[0],self.raw_image.size[1]*scale_radio)
            #self.label_image.setPixmap(jpg)
            #print(self.label_image.width())
            #print(jpg.size())

            imag = cv2.imread(imgName)
            img = cv2.cvtColor(imag,cv2.COLOR_BGR2RGB)
            img = self._scale_image(img, self.logo_size[1])

            #scale_radio = self.logo_size[0] / img.shape[1]
            #dim = (self.logo_size[1],int(img.shape[0] * scale_radio))
            #img = cv2.resize(img, dim,interpolation=cv2.INTER_CUBIC)
            
            img = QtGui.QImage(img,img.shape[1],img.shape[0],img.strides[0],
                               QtGui.QImage.Format_RGB888)
            jpg = QtGui.QPixmap(img)
            self.label_image.setPixmap(jpg)
            imgName = imgName.split('/')[-1]
            self.image_name = imgName
        except:
            pass

    #save upload image to local
    def saveImage(self):  
        fd = './images/' + self.image_name
        self.raw_image.save(fd)
        self.getResult()

    def _scale_image(self,img, long_size):
        if img.shape[0] < img.shape[1]:
            scale = img.shape[1] / long_size
            size = (long_size, math.floor(img.shape[0] / scale))
        else:
            scale = img.shape[0] / long_size
            size = (math.floor(img.shape[1] / scale), long_size)
        return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

    def refine_masksv2(self,masks, labels, im, attrs, attrmap_dict):
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
            refined.append([mask_raw, rle, label, attr_str])

        return refined

    def load_pkl(self,pkl_name):
        with open(pkl_name + '.pkl', 'rb') as f:
            return pickle.load(f)
    
    #get mask result
    def getResult(self):
        img_path = './images/' + self.image_name
        
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.width,self.height), resample = Image.BILINEAR)
        img = F.to_tensor(img)

        self.pred = self.model_test([img.to(device)])[0]
        masks = np.zeros((512,512, len(self.pred['masks'])))
        
        for j,m in enumerate(self.pred['masks']):
            res = transforms.ToPILImage()(m.permute(1,2,0).cpu().numpy())
            res = np.asarray(res.resize((512, 512), resample=Image.BILINEAR))

            masks[:,:,j] = (res[:,:] * 255. > 127).astype(np.uint8)

        labels = self.pred['labels'].cpu().numpy()
        scores = self.pred['scores'].cpu().numpy()
        ascores = self.pred['ascores']

        best_idx = 0
        missing_count = 0
        maskss = []
        labelss = []
        attrs = []

        for _scores in scores:
            if _scores > 0.6:
                best_idx += 1
    
        if best_idx == 0:
            missing_count += 1
            print('no detection')
            return 
    
        if masks.shape[-1]>0:
            imag = cv2.imread(img_path)
            imag = self._scale_image(imag, 1024)
            
            masks = self.refine_masksv2(masks[:,:,:best_idx], labels[:best_idx], imag, ascores[:best_idx], self.attrmap_dict['new2attr'])
            for m, rle, label, attr in masks:
                maskss.append(m)
                labelss.append(label)
                attrs.append(attr)
        else:
            missing_count += 1
            print('no detection')
            return 
    
        if labelss:
            self.classes = labelss
            self.attributes_pred = attrs
            
            draw_masks(img_path, maskss, labelss, imag.shape[1], imag.shape[0],self.save_path)
        else:
            print('no detection')
            return
        self.updateResult()

    # update result to GUI
    def updateResult(self):
        if self.pred != None: 
            total = self.grid.count()
            if total > 3:
                self.deleteItemsOfLayout(self.added)

            pre_path = self.save_path + self.image_name.split('.')[0] + '_pred.jpg'
            
            imag = cv2.imread(pre_path)
            img = cv2.cvtColor(imag,cv2.COLOR_BGR2RGB)
            img = self._scale_image(img, self.logo_size[1])

            #scale_radio = self.logo_size[0] / img.shape[1]
            #dim = (self.logo_size[1],int(img.shape[0] * scale_radio))
            #img = cv2.resize(img, dim,interpolation=cv2.INTER_CUBIC)
            
            img = QtGui.QImage(img,img.shape[1],img.shape[0],img.strides[0],
                               QtGui.QImage.Format_RGB888)
            jpg = QtGui.QPixmap(img)
            #scale_radio = self.logo_size[0]/imag.shape[0]
            #jpg = jpg.scaled(self.logo_size[0],imag.shape[1]*scale_radio)
            #print(self.label_image.width())
            #print(dim)
            #print(jpg.size())

            #jpg = QtGui.QPixmap(pre_path).scaled(self.label_image.width(), self.raw_image.size[1] * scale_radio)
            self.mask_image.setPixmap(jpg)
            self.add()
            
    #dynamically add QLabel
    def add(self):

        font = QtGui.QFont()
        font.setFamily('Arial Black')
        font.setBold(False)
        font.setPointSize(12)
        #font.setWeight(20)

        self.added = QGridLayout()

        visited = []
        count = 1
        
        for i in range(len(self.classes)):
            if self.classes[i] in set(visited):
                count += 1
            else:
                visited.append(self.classes[i])
            self.mask_img = QLabel(self)
            classname = self.categories[self.classes[i]]['name']
            #component_path = './images_pred/' + self.image_name.split('.')[0] + '_'+ classname + '.jpg'
            component_path = './images_pred/' + self.image_name.split('.')[0] + '_'+ classname + str(count) + '.jpg'
        
            component = QtGui.QPixmap(component_path).scaled(64,64)
            self.mask_img.setPixmap(component)
            self.mask_img.resize(64, 64)
            self.added.addWidget(self.mask_img,i,0)

            os.remove(component_path)

            #print(self.dic[self.image_name.split('.')[0]])
            description = str(self.classes[i]) + ' - ' + classname + ": \n"
            #print(len(self.attributes))
            attr = self.attributes_pred[i]

            if attr:
                attr = attr.split(',')
                attr = [int(i) for i in attr]
                if len(attr) > 3:
                    attr = random.sample(attr, 3) 
                #arr = np.array(attr)
                #idxs = arr.argsort()[::-1][0:3]
                for j in range(len(attr)):
                    temp_idx = self.attrmap_dict['attr2new'][attr[j]]
                    if j == 0:
                        temp = self.attributes[temp_idx]['name']
                        description += temp
                    else:
                        temp = self.attributes[temp_idx]['name']
                        description += ', ' + temp 

            self.mask_des = QLabel(self)
            self.mask_des.setText(description)
            self.mask_des.setFont(font)
            self.added.addWidget(self.mask_des,i,1)

        self.grid.addLayout(self.added,1,1)

    #delete dynamic QLabel
    def deleteItemsOfLayout(self,layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
            else:
                deleteItemsOfLayout(item.layout())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())