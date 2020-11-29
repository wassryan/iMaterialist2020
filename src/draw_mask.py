import numpy as np
import json
import colorsys
import random
import cv2
import math

import matplotlib.pyplot as plt

def apply_mask(image, mask, color, classid, alpha=0.4):
    #Apply given class mask to the image.
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])

    return image

def getComponent(prediction,classid,classname,color,alpha,image_name,count):
    image = np.zeros((prediction.shape[0],prediction.shape[1],3))
    for c in range(3):
        image[:,:,c] = np.where(prediction == 1,
                            image[:, :, c] *
                            (1 - alpha) + alpha * color[c] * 255,
                            image[:, :, c])
    plt.cla()
    plt.axis('off')
    plt.imshow(image)
    #component_path = './images/' + self.image_name.split('.')[0] + '_'+ classname + '.jpg'
    component_path = './images_pred/' + image_name+ '_'+ classname + str(count) + '.jpg'
    plt.savefig(component_path,bbox_inches='tight',pad_inches=0.0)
    #return image

def text_image(prediction,class_id):
    text = np.argwhere(prediction == 1)

    text_x = text[int(len(text)/2)][1]

    text_y = text[int(len(text)/2)][0]

    return [text_x,text_y ]

def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def _scale_image(img_path, long_size):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    if img.shape[0] < img.shape[1]:
        scale = img.shape[1] / long_size
        size = (long_size, math.floor(img.shape[0] / scale))
    else:
        scale = img.shape[0] / long_size
        size = (math.floor(img.shape[1] / scale), long_size)
    return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

def load_image(path, shape=(512,512)):
    img = cv2.imread(path) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, shape, interpolation = cv2.INTER_NEAREST)
    return img

def draw_masks(img_path, final_masks, labels,width,height, save_path):
    #image = load_image('images/image0.jpg')
    img_name = img_path.split('/')[-1].split('.')[0]
    # out_name = '/home/lin/Desktop/train/' + img_name + '_pred.jpg' 
    out_name = save_path + img_name + '_pred.jpg' 
    img_toshow = _scale_image(img_path, 1024)
    plt.imshow(img_toshow)
    random_color = random_colors(1)
    img_seg = img_toshow
    plt.cla()
    '''
    # mask each class in different color
    color = random_color[0]
    #masked image
    img_seg = apply_mask(img_seg,final_masks,color,labels)
    text_pos = text_image(final_masks,labels)

    plt.text(text_pos[0], text_pos[1], str(labels), size=11, verticalalignment='top',

                    color='w', backgroundcolor="none",

                    bbox={'facecolor': color, 'alpha': 0.5,

                          'pad': 2, 'edgecolor': 'none'})

    '''
    random_color = random_colors(len(labels))

    with open('label_descriptions.json') as json_f:
        look_up_classes = json.load(json_f)    
        categories = look_up_classes["categories"]

    img_seg = img_toshow
    plt.cla()
    visited = []
    count = 1
    component_list = []
    # draw components
    for i in range(0,len(final_masks)):
        # mask each class in different color
        if labels[i] in set(visited):
            count += 1
        else:
            visited.append(labels[i])
        color = random_color[i]
        #masked image
        img_seg = apply_mask(img_seg,final_masks[i],color,labels[i])
        classname = categories[labels[i]]['name'] 

        getComponent(final_masks[i], labels[i], classname , color, 0.3, img_name,count)
        #component_list.append(component_img)
  
    # draw the predicted image
    plt.cla()
    plt.axis('off')               
    plt.imshow(img_seg)
    for i in range(0,len(final_masks)):
        color = random_color[i]
        classname = categories[labels[i]]['name']
        text_pos = text_image(final_masks[i],labels[i])
        plt.text(text_pos[0], text_pos[1], str(labels[i]), size=11, verticalalignment='top',

                    color='w', backgroundcolor="none",

                    bbox={'facecolor': color, 'alpha': 0.5,

                          'pad': 2, 'edgecolor': 'none'})
    
    plt.savefig(out_name,bbox_inches='tight',pad_inches=0.)
    #return component_list
    #plt.savefig('./test.jpg',bbox_inches='tight',pad_inches=0.0)  

