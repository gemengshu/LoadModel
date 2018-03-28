# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:01:30 2018

@author: Ge Mengshu
"""
from __future__ import print_function
from math import log10
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import UNet
import torchvision

from torchvision.transforms import Compose, CenterCrop, ToTensor, ToPILImage


from PIL import Image

def load_model(data, model_path, cuda = True):

    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")


    unet = UNet()

    if cuda:
        unet = unet.cuda()


    if not cuda:
        unet.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    else:
        unet.load_state_dict(torch.load(model_path))

    if cuda:
        data = Variable(data.cuda())
    else:
        data = Variable(data)
    data = torch.unsqueeze(data, 0)

    output = unet(data)
    if cuda:
       output = output.cuda()

    return output

# image_dir is the full directory of image
# save_dir is the full directory for saving
# model_path is the full directory of trained model
# if GPU is required, then set cuda to TRUE
def TestOnDataset(image_dir, save_dir, model_path, cuda = True):


    filepath = image_dir
    savepath = save_dir

    ori_img = Image.open(filepath).convert('L')

    width, height = ori_img.size
    overlap = 2 # if an image is larger than 1024x1024, then the image will be cut to 4 parts.
                # this parameter represents the width of the overlap region between them.

    # divide the image into four parts
    if width > 1024 or height > 1024:
        crop_size = width//2
        new_img = [ori_img.crop((0,0,crop_size + overlap,crop_size + overlap)),
                   ori_img.crop((crop_size - overlap, 0, width, crop_size + overlap)),
                   ori_img.crop((0, crop_size - overlap, crop_size + overlap, height)),
                   ori_img.crop((crop_size - overlap, crop_size - overlap, width, height))]
    else:
        new_img = [ori_img]

    im_outs = []

    # a flag that denotes whether this image needs padding or not.
    # an image or a region of an image needs padding process
    # if the width or height can't be divided by 4.
    padding_flag = False

    for i in range(0,len(new_img)):
        temp_img = new_img[i]
        transform = ToTensor()
        test = transform(temp_img)
        ori_height = test.size()[1]
        ori_width = test.size()[2]
        if test.size()[1] % 4 or test.size()[2] % 4:
            new_height = np.ceil(float(test.size()[1])/4) * 4 # force the new one's height and width can be divided by 4
            new_width = np.ceil(float(test.size()[2])/4) * 4
            temp = torch.FloatTensor(1, int(new_height), int(new_width)).zero_()
            temp[:, 0 : test.size()[1], 0 : test.size()[2]] = test
            test = temp
            padding_flag = True
        # this function is used to loading model and return the output
        output = load_model(test, model_path, cuda)

        if padding_flag:
            output = output[:,:,0 : ori_height, 0 : ori_width]
            padding_flag = False

        for_save = output[0,:,:,:]

        im_outs.append(for_save)

    to_pil = ToPILImage()

    # this part is for concatenating overlap regions and all parts of images
    if len(im_outs) > 1:
        im_save = Image.new('L', (width,height))

        crop_size = width//2

        if overlap:
            overlap_region1 = torch.max(torch.cat([im_outs[0][:,:-2*overlap,-2*overlap:],
                                         im_outs[1][:,:-2*overlap,:2*overlap]],0) ,0, keepdim = True)
            overlap_region2 = torch.max(torch.cat([im_outs[1][:,-2*overlap:,2*overlap:],
                                         im_outs[3][:,:2*overlap,2*overlap:]],0) ,0, keepdim = True)
            overlap_region3 = torch.max(torch.cat([im_outs[2][:,2*overlap:,-2*overlap:],
                                         im_outs[3][:,2*overlap:,:2*overlap]],0) ,0, keepdim = True)
            overlap_region4 = torch.max(torch.cat([im_outs[0][:,-2*overlap:,:-2*overlap],
                                         im_outs[2][:,:2*overlap,:-2*overlap]],0) ,0, keepdim = True)
            overlap_region_c = torch.max(torch.cat([im_outs[0][:,-2*overlap:,-2*overlap:],
                                         im_outs[1][:,-2*overlap:,:2*overlap],
                                         im_outs[2][:,:2*overlap,:2*overlap],
                                         im_outs[3][:,:2*overlap,-2*overlap:]],0) ,0, keepdim = True)


            im_outs[0][:,:-2*overlap,-2*overlap:] = overlap_region1[0]
            im_outs[0][:,-2*overlap:,:-2*overlap] = overlap_region4[0]
            im_outs[0][:,-2*overlap:,-2*overlap:] = overlap_region_c[0]

            im_outs[1][:,-2*overlap:,2*overlap:] = overlap_region2[0]

            im_outs[2][:,2*overlap:,-2*overlap:] = overlap_region3[0]

        if cuda:
            imout_pil = to_pil(im_outs[0].cpu().data).convert('L')
        else:
            imout_pil = to_pil(im_outs[0].data).convert('L')
        im_save.paste(imout_pil, (0, 0))
        if cuda:
            imout_pil = to_pil(im_outs[1][:,:,2*overlap:].cpu().data).convert('L')
        else:
            imout_pil = to_pil(im_outs[1][:,:,2*overlap:].data).convert('L')
        im_save.paste(imout_pil, (crop_size+overlap, 0))
        if cuda:
            imout_pil = to_pil(im_outs[2][:,2*overlap:,:].cpu().data).convert('L')
        else:
            imout_pil = to_pil(im_outs[2][:,2*overlap:,:].data).convert('L')
        im_save.paste(imout_pil, (0, crop_size+overlap))
        if cuda:
            imout_pil = to_pil(im_outs[3][:,2*overlap:,2*overlap:].cpu().data).convert('L')
        else:
            imout_pil = to_pil(im_outs[3][:,2*overlap:,2*overlap:].data).convert('L')
        im_save.paste(imout_pil, (crop_size+overlap, crop_size+overlap))
    else:
        if cuda:
            im_save = to_pil(im_outs[0].cpu().data).convert('L')
        else:
            im_save = to_pil(im_outs[0].data).convert('L')

    # this part is for saving results.
    # the default result is saving both original image and result in one image.
    # if only the result is needed, then modifications can be done here.
    # ori_img is the original one and im_save is the final result.
    cat_image = Image.new('L', (width * 2, height))
    cat_image.paste(ori_img, (0,0))
    cat_image.paste(im_save, (width, 0))
    cat_image.save(savepath)

# image directory
dir = "D:/GMS/Documents/BNL/load_model_code/"

# saving directory
save_dir = "D:/GMS/Documents/BNL/load_model_code/test/"

# model path
model_out_path = "D:/GMS/Documents/BNL/20180208/lr001_weightdecay00001.pth"

count = 1;

for root, dirs, files in os.walk(dir): #
    for filename in files:
        if filename.endswith(('.png','.tif','.jpg')): # supported formats of the images

            image_name = filename
            filepath = os.path.join(root, image_name)
            savepath = os.path.join(save_dir, image_name)

            TestOnDataset(filepath, savepath, model_out_path, cuda = False)

            print("Processing {} images\n".format(count))

            count = count + 1
