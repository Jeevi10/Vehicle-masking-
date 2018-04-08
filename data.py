#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 14:09:22 2017

@author: oem
"""
from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread
from skimage.transform import resize

data_path = "/home/jeev/Documents/project598-520/"

image_rows = 192
image_cols = 192


def create_train_data():
    train_data_path = os.path.join(data_path, 'train_hq')
    train_datamask_path =os.path.join(data_path, 'train_masks')
    images_train = os.listdir(train_data_path)
    imagemask_train =os.listdir(train_datamask_path)
    total_img = len(images_train)
    total_mask= len(imagemask_train)
    
    

    imgs = np.ndarray((int(total_img), image_rows, image_cols,3), dtype=np.uint8)
    imgs_mask = np.ndarray((int(total_mask), image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images_train:
        image_mask_name = image_name.split('.')[0] + '_mask.gif'
        img = imread(os.path.join(train_data_path, image_name), as_grey=False)
    
        img_mask = imread(os.path.join(train_datamask_path, image_mask_name), as_grey=True)
        img = resize(img, (128, 128,3), preserve_range=True)
        img_mask = resize(img_mask, (128, 128), preserve_range=True)
        
        img = np.array([img])
        img_mask = np.array([img_mask])
        
        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:

            print('Done: {0}/{1} images'.format(i, total_img))
        i += 1
    print('Loading done.')

    np.save('imgs_traint.npy', imgs)
    np.save('imgs_mask_traint.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_traint.npy')
    imgs_mask_train = np.load('imgs_mask_traint.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    test_data_path = os.path.join(data_path, 'outtestset')
    #test_mask_path = os.path.join(data_path, 'testmsk')
    images = os.listdir(test_data_path)
   # masks = os.listdir(test_mask_path)
    total_img = len(images)
    #total_imgs =len(masks)

    imgs = np.ndarray((int(total_img), 128, 128,3), dtype=np.uint8)
    #imgs_mask=np.ndarray((int(total_img),128,128),dtype=np.uint8)
    imgs_id = np.ndarray((total_img, ),dtype=object)
    

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id1 =str(image_name.split(".")[0])
              
        #print(img_id1)
        #img_id =image_name.split(".")[0]+'_mask.gif'
        
        img = imread(os.path.join(test_data_path, image_name), as_grey=False)
        #mask= imread(os.path.join(test_mask_path, img_id), as_grey=True)
        img = resize(img, (128, 128,3), preserve_range=True)
        #mask = resize(mask, (128, 128), preserve_range=True)

        img = np.array([img])
       # mask=np.array([mask])


        imgs[i] = img
        #imgs_mask[i]=mask
        imgs_id[i] = img_id1
        
      
       

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total_img))
        i += 1
    print('Loading done.')

    np.save('imgs_test1.npy', imgs)
    #np.save('imgsmask_test.npy',imgs_mask)
    np.save('imgs_id_test1.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test1.npy')
    #imgs_test_mask=np.load('imgsmask_test.npy')
    imgs_id = np.load('imgs_id_test1.npy')
    return imgs_test,imgs_id

if __name__ == '__main__':
    create_test_data()
    #create_test_data()

