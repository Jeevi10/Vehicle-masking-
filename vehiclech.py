#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:50:48 2017

@author: jeev
"""

#%%
from __future__ import print_function
import numpy as np
import tensorflow as tf

from skimage.transform import resize
from skimage.io import imsave


from six.moves import cPickle as pickle
import tensorflow.contrib.slim as slim
from six.moves import range
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import time
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from scipy.misc import imresize
from keras.layers import Dense, Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose,Dropout

from data import load_train_data,load_test_data
import os, glob, math
import sys
from data import load_test_data,load_train_data
from sklearn.model_selection import train_test_split
from keras import backend as k
from keras.models import Sequential, Model
from keras.optimizers import Adam
from functools import reduce
 






image_size=700;
number_channels=3

tensor_path="/home/jeev/Documents/project598-520/tensor/"
data_dir = "/home/jeev/Documents/project598-520/train_hq/"
mask_dir = "/home/jeev/Documents/project598-520/train_masks/"

all_images = os.listdir(data_dir)
train_images, validation_images = train_test_split(all_images, train_size=0.8, test_size=0.2)


def grey2rgb(img):
    new_img = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img.append(list(img[i][j])*3)
    new_img = np.array(new_img).reshape(img.shape[0], img.shape[1], 3)
    return new_img




def data_gen_small(data_dir, mask_dir, images, batch_size, dims):
        """
        data_dir: where the actual images are kept
        mask_dir: where the actual masks are kept
        images: the filenames of the images we want to generate batches from
        batch_size: self explanatory   tf.summary.scalar("accuracy",dicecoff_train)
        
        dims: the dimensions in which we want to rescale our images
        """
        while True:
            ix = np.random.choice(np.arange(len(images)), batch_size)
            imgs = []
            labels = []
            for i in ix:
                # images
                original_img = load_img(data_dir + images[i])
                resized_img = imresize(original_img, dims+[3])
                array_img = img_to_array(resized_img)/255
                imgs.append(array_img)
                
                # masks
                original_mask = load_img(mask_dir + images[i].split(".")[0] + '_mask.gif')
                resized_mask = imresize(original_mask, dims+[3])
                array_mask = img_to_array(resized_mask)/255
                labels.append(array_mask[:, :, 0])
            imgs = np.array(imgs)
            labels = np.array(labels)
           
            yield imgs, labels.reshape(-1, dims[0], dims[1], 1)
            
            

            





            
            
            

train_gen= data_gen_small(data_dir,mask_dir,train_images,3,[128,128])
vali_gen=data_gen_small(data_dir,mask_dir,validation_images,3,[128,128])
imgs,labels =next(train_gen)

plt.imshow(imgs[0])
plt.show()
plt.imshow(grey2rgb(labels[0]), alpha=0.5) 
plt.show()


def val_gene(vali_gen):
    val_imgs,val_labels=next(vali_gen)
    
    return val_imgs,val_labels
  



def dice_coef(y_true, y_pred):
    print(np.shape(y_true),np.shape(y_pred))
    """Compute the mean(batch-wise) of dice coefficients"""
    md=tf.constant(0.0)
    y_true=tf.cast(y_true, tf.float32)
    y_pred=tf.cast(y_pred,tf.float32)
    y_true_f = slim.flatten(y_true)
    y_pred_f = slim.flatten(y_pred)
    for i in range(batch_size):
        union=tf.reduce_sum(y_true_f[i]) + tf.reduce_sum(y_pred_f[i]) 
        md = tf.cond(tf.equal(union,0.0), lambda: tf.add(md,1.0),lambda: tf.add(md,tf.div(2.*tf.reduce_sum(tf.multiply(y_true_f[i],y_pred_f[i])),union))) 
   
    return tf.div(md,batch_size)

def accuracy(y_true,y_pred):
    smooth =0.001
    y_true=tf.cast(y_true, tf.float32)
    y_pred=tf.cast(y_pred,tf.float32)
    y_true_f = slim.flatten(y_true)
    y_pred_f = slim.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def conv_layer(X,W_shape,b_shape,name,padding="SAME"):
    W = weight_variables(W_shape)
    b = bias_variables(b_shape)
    
    return tf.nn.relu(tf.nn.conv2d(X,W,strides=[1,1,1,1],padding=padding)+b)


def weight_variables(shape):
    initial =tf.truncated_normal(shape,stddev=0.1)
    
    return tf.Variable(initial)

def bias_variables(shape):
    initial =tf.zeros(shape)
    return tf.Variable(initial)

def deconv_layer(X,W_shape,b_shape,name,padding ="SAME"):
    W =weight_variables(W_shape)
    b=bias_variables(b_shape)
    
    X_shape =tf.shape(X)
    out_shape=tf.stack([X_shape[0],X_shape[1],X_shape[2],W_shape[2]])
    
    return tf.nn.conv2d_transpose(X,W,out_shape,[1,1,1,1],padding=padding)+b



def down(input_layer, filters, pool=True,name="conv"):
    with tf.name_scope(name):
        conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(input_layer)
        residual = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
        if pool:
            max_pool = MaxPool2D()(residual)
            return max_pool, residual
        else:
            return residual

def up(input_layer, residual, filters,name="upconv"):
    with tf.name_scope(name):
        filters=int(filters)
        upsample = UpSampling2D()(input_layer)
        upconv = Conv2D(filters, kernel_size=(2, 2), padding="same")(upsample)
        concat = Concatenate(axis=3)([residual, upconv])
        conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(concat)
        conv2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
        conv2
        return conv2         



graph =tf.Graph();


batch_size = 3
patch_size = 3

num_channels=3

image_width=128
image_hight=128



with graph.as_default():
    
    

    tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size,image_width,image_hight,num_channels),name="input")
    tf_train_lables = tf.placeholder(tf.float32,shape=(batch_size,image_width,image_hight,1),name="Labels")
    
    
  
    
    
    
    
    def inference(data):
        
        filters = 32
        input_layer = data
        layers = [input_layer]
        residuals = []
        
        # Down 1, 256
        d0,res0 =down(input_layer,filters,"conv1")
        residuals.append(res0)
       
        filters *= 2
        
        
        #Down 2,128
        d1, res1 = down(d0, filters,"conv2")
        residuals.append(res1)
        
        
        filters *= 2
        
        
        # Down 2, 64
        d2, res2 = down(d1, filters,"conv3")
        residuals.append(res2)
        
        
        filters *= 2
        
        # Down 3, 32
        d3, res3 = down(d2, filters,"conv4")
        residuals.append(res3)
        
        
        filters *= 2
        
        # Down 4, 16
        d4, res4 = down(d3, filters,"conv5")
        residuals.append(res4)
        
        
        filters *= 2
        
        #d5,res5 =down(d4,filters)
        #residuals.append(res5)
        # Down 5, 8   
        #filters *= 2
        
        d5 = down(d4, filters, pool=False,name="Upconv")
        
        #up0 =up(d6,residual=residuals[-1],filters=filters/2)
        
        # Up 1, 16
        up1 = up(d5, residual=residuals[-1], filters=filters/2,name="upconv1")
        
        
        filters /= 2
        
        # Up 2,  32
        up2 = up(up1, residual=residuals[-2], filters=filters/2,name="upconv2")
    
        
        filters /= 2
        
        # Up 3, 64
        up3 = up(up2, residual=residuals[-3], filters=filters/2,name="upconv3")
       
        filters /= 2
        
        # Up 4, 128
        up4 = up(up3, residual=residuals[-4], filters=filters/2,name="upconv4")   
        
        filters /= 2
        # Up 5 ,256
        up5 =up(up4,residual=residuals[-5],filters=filters/2,name="upconv5")
        
        
        out = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(up5)
        
        
  
        return out
    
    
    with tf.device('/gpu:0'):
       
        models=inference(tf_train_dataset)
        print("modshape"+str(np.shape(models)))
        
        cross_entropy=tf.contrib.keras.backend.binary_crossentropy(models,tf_train_lables)
    
        with tf.name_scope("loss"):       
            loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            
        with tf.name_scope("train_step"):
            train_step = tf.train.AdamOptimizer(learning_rate=0.0001,beta1=0.9,beta2=0.999,epsilon=1e-08,use_locking =False).minimize(loss)
     
        with tf.name_scope("accura"):
            dicecoff_train=accuracy(tf_train_lables,models)  
        
    
        
    
num_steps =4000 
#saver=tf.train.Saver()

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
   # save_path=saver.save(session,tensor_path,global_step=2)
  
    Writer=tf.summary.FileWriter('/home/jeev/Documents/project598-520/trainingpara1/train/plot_train')
    Writerval=tf.summary.FileWriter("/home/jeev/Documents/project598-520/trainingpara1/train/plot_val")

    tf.summary.scalar("Loss",loss)
    tf.summary.scalar("accuracy",dicecoff_train)
    merged_summary=tf.summary.merge_all()
  
   # Writer.add_graph(session.graph)
    #Writerval.add_graph(session.graph)
    avgloss=[]
    avgaccuracy=[]
    valloss=[]
    valacc1=[]
    for step in range(num_steps):
        imgs,labels=next(train_gen)        
        feed_dict1={tf_train_dataset :imgs,tf_train_lables:labels}       
        _,m,l,d=session.run([train_step,models,loss,dicecoff_train],feed_dict=feed_dict1)
       # save_path=saver.save(session,tensor_path+"/temp/model.ckpt")
        s=session.run(merged_summary,feed_dict=feed_dict1)
        Writer.add_summary(s,step)
        Writer.flush()
        val_imgs,val_lab=next(vali_gen)
        feed_dict2={tf_train_dataset :val_imgs,tf_train_lables:val_lab}  
        v,valacc,vloss=session.run([models,dicecoff_train,loss],feed_dict=feed_dict2)
        s1=session.run(merged_summary,feed_dict=feed_dict2)
        Writerval.add_summary(s1,step)
        Writerval.flush()
        avgloss.append(l)
        avgaccuracy.append(d)
        
        
        if (step % 100==0):
#            val_imgs,val_lab=next(vali_gen)
#            feed_dict2={tf_train_dataset :val_imgs,tf_train_lables:val_lab}  
#            v,valacc,vloss=session.run([models,dicecoff_train,loss],feed_dict=feed_dict2)
            valloss.append(vloss)
            valacc1.append(valacc)
            print('Minibatch loss at ' +str(reduce( lambda x,y:x+y,avgloss)/len(avgloss)))
            print('Minibatch accuracy at ' +str(reduce( lambda x,y:x+y,avgaccuracy)/len(avgaccuracy)))
            print('Minibatch val accuracy at ' +str(reduce( lambda x,y:x+y,valacc1)/len(valacc1)))
            print('Valloss'+str(reduce( lambda x,y:x+y,valloss)/len(valloss)))
            
            imgo=tf.slice(m,[1,0,0,0],[1,128,128,1])
            v=tf.slice(v,[1,0,0,0],[1,128,128,1])
            print(np.shape(imgo))
            
            imgo=tf.squeeze(imgo,[3])
            imgo=tf.squeeze(imgo,[0])
            print(np.min(imgo.eval(session=session)))
            print(np.max(imgo.eval(session=session)))
            v=tf.squeeze(v,[3])
            v=tf.squeeze(v,[0])
            
            imgs0=tf.squeeze(labels[1],[2])
          
            print(np.shape(imgo))
            
#            plt.imshow(car.eval(session=session))
#            plt.show()
            plt.imshow(imgo.eval(session=session))
            plt.show()
            plt.imshow(imgs0.eval(session=session))
            plt.show()
            plt.imshow(v.eval(session=session))
            plt.show()
            
           