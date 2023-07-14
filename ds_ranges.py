# library imports
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import pickle
import random
import copy
import itertools
from PIL import Image, ImageEnhance
# import cv2
import mxnet as mx
from gluoncv import data, utils
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, AnchoredText
from matplotlib.transforms import blended_transform_factory
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm

import mean_ap

lst = ["saturation", "brightness", "hue", "contrast", "noise"]
permutations = list(itertools.permutations(lst))

def create_dataset_range_saturation (small_dataset, n = 11):
    
    saturation_range = tf.linspace(-1.0, 1.0, n)
    ds_range = []
    
    conv_dataset = []
    
    for j in range(len(small_dataset)):
        conv_dataset_j = []
        conv_dataset_j.append(tf.image.convert_image_dtype(tf.constant(small_dataset[j][0].asnumpy()), dtype=tf.float32))
        conv_dataset_j.append(small_dataset[j][1])
        conv_dataset.append(conv_dataset_j)
    
    
    for i in range(n):
        ds_from_range = copy.deepcopy(conv_dataset)
        for k in range(len(conv_dataset)):
            img = conv_dataset[k][0]
            image_hsv = tf.image.rgb_to_hsv(img)
            image_hsv_adjusted = tf.stack([
                image_hsv[..., 0],
                tf.clip_by_value(image_hsv[..., 1] + saturation_range[i], 0.0, 1.0),
                image_hsv[..., 2]
            ], axis=-1)
            
            image_rgb_adjusted = tf.image.hsv_to_rgb(image_hsv_adjusted)
            ds_from_range[k][0] = mx.nd.array(np.round(image_rgb_adjusted.numpy()*255)).astype("uint8")
            
        ds_range.append(ds_from_range)
        
    return ds_range

def create_dataset_range_saturation_up (small_dataset, n = 11):
    
    saturation_range = tf.linspace(0.0, 1.0, n)
    ds_range = []
    
    conv_dataset = []
    
    for j in range(len(small_dataset)):
        conv_dataset_j = []
        conv_dataset_j.append(tf.image.convert_image_dtype(tf.constant(small_dataset[j][0].asnumpy()), dtype=tf.float32))
        conv_dataset_j.append(small_dataset[j][1])
        conv_dataset.append(conv_dataset_j)
    
    
    for i in range(n):
        ds_from_range = copy.deepcopy(conv_dataset)
        for k in range(len(conv_dataset)):
            img = conv_dataset[k][0]
            image_hsv = tf.image.rgb_to_hsv(img)
            image_hsv_adjusted = tf.stack([
                image_hsv[..., 0],
                tf.clip_by_value(image_hsv[..., 1] + saturation_range[i], 0.0, 1.0),
                image_hsv[..., 2]
            ], axis=-1)
            
            image_rgb_adjusted = tf.image.hsv_to_rgb(image_hsv_adjusted)
            ds_from_range[k][0] = mx.nd.array(np.round(image_rgb_adjusted.numpy()*255)).astype("uint8")
            
        ds_range.append(ds_from_range)
        
    return ds_range

def create_dataset_range_saturation_down (small_dataset, n = 11):
    
    saturation_range = tf.linspace(0.0, -1.0, n)
    ds_range = []
    
    conv_dataset = []
    
    for j in range(len(small_dataset)):
        conv_dataset_j = []
        conv_dataset_j.append(tf.image.convert_image_dtype(tf.constant(small_dataset[j][0].asnumpy()), dtype=tf.float32))
        conv_dataset_j.append(small_dataset[j][1])
        conv_dataset.append(conv_dataset_j)
    
    
    for i in range(n):
        ds_from_range = copy.deepcopy(conv_dataset)
        for k in range(len(conv_dataset)):
            img = conv_dataset[k][0]
            image_hsv = tf.image.rgb_to_hsv(img)
            image_hsv_adjusted = tf.stack([
                image_hsv[..., 0],
                tf.clip_by_value(image_hsv[..., 1] + saturation_range[i], 0.0, 1.0),
                image_hsv[..., 2]
            ], axis=-1)
            
            image_rgb_adjusted = tf.image.hsv_to_rgb(image_hsv_adjusted)
            ds_from_range[k][0] = mx.nd.array(np.round(image_rgb_adjusted.numpy()*255)).astype("uint8")
            
        ds_range.append(ds_from_range)
        
    return ds_range

def create_dataset_range_brightness_up (small_dataset, n=11):
    
    brightness_range = tf.linspace(0.0, 1.0, n)
    ds_range = []
    
    conv_dataset = []
    
    for j in range(len(small_dataset)):
        conv_dataset_j = []
        conv_dataset_j.append(tf.image.convert_image_dtype(tf.constant(small_dataset[j][0].asnumpy()), dtype=tf.float32))
        conv_dataset_j.append(small_dataset[j][1])
        conv_dataset.append(conv_dataset_j)
        
    for i in range(n):
        ds_from_range = copy.deepcopy(conv_dataset)
        for k in range(len(conv_dataset)):
            img = conv_dataset[k][0]
            image_hsv = tf.image.rgb_to_hsv(img)
            image_hsv_adjusted = tf.stack([
                image_hsv[..., 0],
                image_hsv[..., 1],
                tf.clip_by_value(image_hsv[..., 2] + brightness_range[i], 0.0, 1.0)
            ], axis=-1)
            
            image_rgb_adjusted = tf.image.hsv_to_rgb(image_hsv_adjusted)
            ds_from_range[k][0] = mx.nd.array(np.round(image_rgb_adjusted.numpy()*255)).astype("uint8")
            
        ds_range.append(ds_from_range)
        
    return ds_range

def create_dataset_range_brightness_down (small_dataset, n=11):
    
    brightness_range = tf.linspace(0.0, -1.0, n)
    ds_range = []
    
    conv_dataset = []
    
    for j in range(len(small_dataset)):
        conv_dataset_j = []
        conv_dataset_j.append(tf.image.convert_image_dtype(tf.constant(small_dataset[j][0].asnumpy()), dtype=tf.float32))
        conv_dataset_j.append(small_dataset[j][1])
        conv_dataset.append(conv_dataset_j)
        
    for i in range(n):
        ds_from_range = copy.deepcopy(conv_dataset)
        for k in range(len(conv_dataset)):
            img = conv_dataset[k][0]
            image_hsv = tf.image.rgb_to_hsv(img)
            image_hsv_adjusted = tf.stack([
                image_hsv[..., 0],
                image_hsv[..., 1],
                tf.clip_by_value(image_hsv[..., 2] + brightness_range[i], 0.0, 1.0)
            ], axis=-1)
            
            image_rgb_adjusted = tf.image.hsv_to_rgb(image_hsv_adjusted)
            ds_from_range[k][0] = mx.nd.array(np.round(image_rgb_adjusted.numpy()*255)).astype("uint8")
            
        ds_range.append(ds_from_range)
        
    return ds_range

def create_dataset_range_hue (small_dataset, n=11):
    
    hue_range = tf.linspace(0.0, 1.0, n)
    ds_range = []
    
    conv_dataset = []
    
    for j in range(len(small_dataset)):
        conv_dataset_j = []
        conv_dataset_j.append(tf.image.convert_image_dtype(tf.constant(small_dataset[j][0].asnumpy()), dtype=tf.float32))
        conv_dataset_j.append(small_dataset[j][1])
        conv_dataset.append(conv_dataset_j)
        
    for i in range(n):
        ds_from_range = copy.deepcopy(conv_dataset)
        for k in range(len(conv_dataset)):
            img = conv_dataset[k][0]
            image_hsv = tf.image.rgb_to_hsv(img)
            image_hsv_adjusted = tf.stack([
                tf.math.mod(image_hsv[..., 0] + hue_range[i], 1),
                image_hsv[..., 1],
                image_hsv[..., 2]
            ], axis=-1)
            
            image_rgb_adjusted = tf.image.hsv_to_rgb(image_hsv_adjusted)
            ds_from_range[k][0] = mx.nd.array(np.round(image_rgb_adjusted.numpy()*255)).astype("uint8")
            
        ds_range.append(ds_from_range)
        
    return ds_range

def create_dataset_range_contrast_up (small_dataset, n=11):
    
    contrast_range_up = np.logspace(math.log(1, 10), math.log(30, 10), n)
    ds_range = []
    
    conv_dataset = []
    
    for j in range(len(small_dataset)):
        conv_dataset_j = []
        conv_dataset_j.append(small_dataset[j][0].asnumpy())
        conv_dataset_j.append(small_dataset[j][1])
        conv_dataset.append(conv_dataset_j)
        
    for i in range(n):
        ds_from_range = copy.deepcopy(conv_dataset)
        for k in range(len(conv_dataset)):
            img = conv_dataset[k][0]
            img = tf.image.adjust_contrast(img, contrast_range_up[i])
            ds_from_range[k][0] = mx.nd.array(img.numpy()).astype("uint8")
            
        ds_range.append(ds_from_range)
        
    return ds_range

def create_dataset_range_contrast_down (small_dataset, n=11):
    
    contrast_range_down = np.linspace(1.0, 0.0, n)
    ds_range = []
    
    conv_dataset = []
    
    for j in range(len(small_dataset)):
        conv_dataset_j = []
        conv_dataset_j.append(small_dataset[j][0].asnumpy())
        conv_dataset_j.append(small_dataset[j][1])
        conv_dataset.append(conv_dataset_j)
        
    for i in range(n):
        ds_from_range = copy.deepcopy(conv_dataset)
        for k in range(len(conv_dataset)):
            img = conv_dataset[k][0]
            img = tf.image.adjust_contrast(img, contrast_range_down[i])
            ds_from_range[k][0] = mx.nd.array(img.numpy()).astype("uint8")
            
        ds_range.append(ds_from_range)
        
    return ds_range

def create_dataset_range_noise (small_dataset, n=11):
    
    noise_intensity = tf.linspace(0.0, 0.3, n)
    std = tf.linspace(0.0, 2.0, n)
    salt_pepper_ratio = tf.linspace(0.0, 0.1, n)
    ds_range = []
    
    conv_dataset = []
    
    for j in range(len(small_dataset)):
        conv_dataset_j = []
        conv_dataset_j.append(tf.image.convert_image_dtype(tf.constant(small_dataset[j][0].asnumpy()), dtype=tf.float32))
        conv_dataset_j.append(small_dataset[j][1])
        conv_dataset.append(conv_dataset_j)
        
    for i in range(n):
        ds_from_range = copy.deepcopy(conv_dataset)
        for k in range(len(conv_dataset)):
            img = conv_dataset[k][0]
            gaussian_noise = tf.random.normal(shape=img.shape, mean=0.0, stddev=std[i])
            salt_pepper_noise = tf.random.uniform(shape=img.shape, minval=0, maxval=1)
            salt_noise = tf.cast(salt_pepper_noise < salt_pepper_ratio[i] / 2, dtype=tf.float32) * 1.0
            pepper_noise = tf.cast(salt_pepper_noise > 1 - salt_pepper_ratio[i] / 2, dtype=tf.float32) * -1.0
            noisy_image = img + gaussian_noise * noise_intensity[i] + salt_noise + pepper_noise
            noisy_image = tf.clip_by_value(noisy_image, 0, 1)

            ds_from_range[k][0] = mx.nd.array(np.round(noisy_image.numpy()*255)).astype("uint8")
            
        ds_range.append(ds_from_range)
        
    return ds_range

def add_saturation_to_dataset(small_dataset, saturation_level=1):
    
    appllied_saturation = tf.linspace(0.0, 1.0, 11)[saturation_level]
    
    conv_dataset = []
    
    for j in range(len(small_dataset)):
        conv_dataset_j = []
        conv_dataset_j.append(tf.image.convert_image_dtype(tf.constant(small_dataset[j][0].asnumpy()), dtype=tf.float32))
        conv_dataset_j.append(small_dataset[j][1])
        conv_dataset.append(conv_dataset_j)
        
    for k in range(len(conv_dataset)):
        img = conv_dataset[k][0]
        image_hsv = tf.image.rgb_to_hsv(img)
        image_hsv_adjusted = tf.stack([
            image_hsv[..., 0],
            tf.clip_by_value(image_hsv[..., 1] + appllied_saturation, 0.0, 1.0),
            image_hsv[..., 2]
        ], axis=-1)

        image_rgb_adjusted = tf.image.hsv_to_rgb(image_hsv_adjusted)
        conv_dataset[k][0] = mx.nd.array(np.round(image_rgb_adjusted.numpy()*255)).astype("uint8")
        
    return conv_dataset

def add_brightness_to_dataset(small_dataset, brightness_level=1):
    
    appllied_brightness = tf.linspace(0.0, 1.0, 11)[brightness_level]
    
    conv_dataset = []
    
    for j in range(len(small_dataset)):
        conv_dataset_j = []
        conv_dataset_j.append(tf.image.convert_image_dtype(tf.constant(small_dataset[j][0].asnumpy()), dtype=tf.float32))
        conv_dataset_j.append(small_dataset[j][1])
        conv_dataset.append(conv_dataset_j)
        
    for k in range(len(conv_dataset)):
        img = conv_dataset[k][0]
        image_hsv = tf.image.rgb_to_hsv(img)
        image_hsv_adjusted = tf.stack([
            image_hsv[..., 0],
            image_hsv[..., 1],
            tf.clip_by_value(image_hsv[..., 2] + appllied_brightness, 0.0, 1.0)
        ], axis=-1)

        image_rgb_adjusted = tf.image.hsv_to_rgb(image_hsv_adjusted)
        conv_dataset[k][0] = mx.nd.array(np.round(image_rgb_adjusted.numpy()*255)).astype("uint8")
        
    return conv_dataset

def add_hue_to_dataset(small_dataset, hue_level=1):
    
    appllied_hue = tf.linspace(0.0, 1.0, 12)[hue_level]
    
    conv_dataset = []
    
    for j in range(len(small_dataset)):
        conv_dataset_j = []
        conv_dataset_j.append(tf.image.convert_image_dtype(tf.constant(small_dataset[j][0].asnumpy()), dtype=tf.float32))
        conv_dataset_j.append(small_dataset[j][1])
        conv_dataset.append(conv_dataset_j)
        
    for k in range(len(conv_dataset)):
        img = conv_dataset[k][0]
        image_hsv = tf.image.rgb_to_hsv(img)
        image_hsv_adjusted = tf.stack([
            tf.math.mod(image_hsv[..., 0] + appllied_hue, 1),
            image_hsv[..., 1],
            image_hsv[..., 2]
        ], axis=-1)

        image_rgb_adjusted = tf.image.hsv_to_rgb(image_hsv_adjusted)
        conv_dataset[k][0] = mx.nd.array(np.round(image_rgb_adjusted.numpy()*255)).astype("uint8")
        
    return conv_dataset

def add_contrast_to_dataset(small_dataset, contrast_level=1):
    
    appllied_contrast = np.logspace(math.log(1, 10), math.log(30, 10), 11)[contrast_level]
    
    conv_dataset = []
        
    for j in range(len(small_dataset)):
        conv_dataset_j = []
        conv_dataset_j.append(small_dataset[j][0].asnumpy())
        conv_dataset_j.append(small_dataset[j][1])
        conv_dataset.append(conv_dataset_j)
        
    for k in range(len(conv_dataset)):
        img = conv_dataset[k][0]
        img = tf.image.adjust_contrast(img, appllied_contrast)
        conv_dataset[k][0] = mx.nd.array(img.numpy()).astype("uint8")
        
    return conv_dataset

def add_noise_to_dataset(small_dataset, noise_level=1):
    
    noise_intensity = tf.linspace(0.0, 0.3, 11)[noise_level]
    std = tf.linspace(0.0, 2.0, 11)[noise_level]
    salt_pepper_ratio = tf.linspace(0.0, 0.1, 11)[noise_level]
    
    conv_dataset = []
        
    for j in range(len(small_dataset)):
        conv_dataset_j = []
        conv_dataset_j.append(tf.image.convert_image_dtype(tf.constant(small_dataset[j][0].asnumpy()), dtype=tf.float32))
        conv_dataset_j.append(small_dataset[j][1])
        conv_dataset.append(conv_dataset_j)
        
    for k in range(len(conv_dataset)):
        img = conv_dataset[k][0]
        gaussian_noise = tf.random.normal(shape=img.shape, mean=0.0, stddev=std)
        salt_pepper_noise = tf.random.uniform(shape=img.shape, minval=0, maxval=1)
        salt_noise = tf.cast(salt_pepper_noise < salt_pepper_ratio / 2, dtype=tf.float32) * 1.0
        pepper_noise = tf.cast(salt_pepper_noise > 1 - salt_pepper_ratio / 2, dtype=tf.float32) * -1.0
        noisy_image = img + gaussian_noise * noise_intensity + salt_noise + pepper_noise
        noisy_image = tf.clip_by_value(noisy_image, 0, 1)

        conv_dataset[k][0] = mx.nd.array(np.round(noisy_image.numpy()*255)).astype("uint8")
        
    return conv_dataset

def create_dataset_range_permutations(small_dataset, distortion_level=1):
    
    lst = ["saturation", "brightness", "hue", "contrast", "noise"]
    permutations = list(itertools.permutations(lst))
    
    ds_range = []
    
    for permutation in permutations:
        distorted_dataset = copy.deepcopy(small_dataset)
        
        for step in permutation:
            if step == "saturation":
                distorted_dataset = add_saturation_to_dataset(distorted_dataset, saturation_level=distortion_level)
            elif step == "brightness":
                distorted_dataset = add_brightness_to_dataset(distorted_dataset, brightness_level=distortion_level)
            elif step == "hue":
                distorted_dataset = add_hue_to_dataset(distorted_dataset, hue_level=distortion_level)
            elif step == "contrast":
                distorted_dataset = add_contrast_to_dataset(distorted_dataset, contrast_level=distortion_level)
            elif step == "noise":
                distorted_dataset = add_noise_to_dataset(distorted_dataset, noise_level=distortion_level)
                
        ds_range.append(distorted_dataset)
    
    return ds_range

def visualise_ds_range(dataset_range, n=0):
    
    shape = dataset_range[0][n][0].asnumpy().shape
    img_width = shape[1]
    img_height = shape[0]
    side_ratio = img_width / img_height
    
    distortion_type = input("Enter distortion type: ")
    
    ncols = 4
    if distortion_type != "permutations":
        nrows = math.ceil(len(dataset_range)/ncols)
    else:
        nrows = math.ceil((len(dataset_range) / 10) / ncols)    
    
    
    if distortion_type == "noise":
        fig = plt.figure(figsize=(side_ratio*5*ncols, 6.03*nrows))
    else:
        fig = plt.figure(figsize=(side_ratio*5*ncols, 5.17*nrows))
    
    contrast_factors = [round(element, 2) for element in np.logspace(math.log(1, 10), math.log(50, 10), 11)]

    for i, dataset in enumerate(dataset_range):
        if distortion_type != "permutations":
            plt.subplot(nrows, ncols, i+1)
            plt.imshow(dataset[n][0].asnumpy())
            plt.axis("off")
        else:
            if i % 10 == 0:
                plt.subplot(nrows, ncols, (i//10)+1)
                plt.imshow(dataset[n][0].asnumpy())
                plt.axis("off")
                plt.title(str(i), y=-0.085, pad=0.1, fontsize=36)
            else:
                pass
                
        if distortion_type == "saturation up" or distortion_type == "brightness up":
            plt.title("+" + str(i*10), y=-0.1, pad=0.1, fontsize=36)
        elif distortion_type == "saturation down" or distortion_type == "brightness down":
            plt.title("-" + str(i*10), y=-0.1, pad=0.1, fontsize=36)
        elif distortion_type == "hue":
            plt.title(str(round(i*36)) + "°", y=-0.1, pad=0.1, fontsize=36)
        elif distortion_type == "contrast up":
            plt.title(str(contrast_factors[i]), y=-0.1, pad=0.1, fontsize=36)
        elif distortion_type == "contrast down":
            plt.title(str(round(1 - i/10, 1)), y=-0.1, pad=0.1, fontsize=36)
        elif distortion_type == "noise":
            plt.title(str(round(i*3/100, 2)) + "\n" + str(round(i/5, 2)) + "\n" + str(round(i/100, 2)), y=-0.305, pad=0.1, fontsize=28)
            
    
    if distortion_type == "saturation up" or distortion_type == "saturation down":
        plt.suptitle("Saturation Increment (HSV)", fontsize=42, y=1.01)
        plt.tight_layout()
    if distortion_type == "brightness up" or distortion_type == "brightness down":
        plt.suptitle("Brightness (=Value) Increment (HSV)", fontsize=42, y=1.01)
        plt.tight_layout()
    if distortion_type == "hue":
        plt.suptitle("Hue Roation (HSV)", fontsize=42, y=1.01)
        plt.tight_layout()
    if distortion_type == "contrast up" or distortion_type == "contrast down":
        plt.suptitle("`contrast_factor` Parameter `of `tf.image.adjust_contrast()`", fontsize=42, y=1.01)
        plt.tight_layout()
    if distortion_type == "noise":
        plt.suptitle("noise_intensity\nstd_dev\nsalt_pepper_ratio", fontsize=34, y=0.955)
        plt.subplots_adjust(wspace=0.07)
    if distortion_type == "permutations":
        plt.suptitle("Permutation Index", fontsize=42, y=1.01)
        plt.tight_layout()
    
    plt.show();
    
    return fig

def visualise_distortion_types(dataset_range_list, n=0):
    
    shape = dataset_range_list[0][0][n][0].asnumpy().shape
    img_width = shape[1]
    img_height = shape[0]
    side_ratio = img_width / img_height
    
    image_list = []
    for dataset_range in dataset_range_list:
        for i, dataset in enumerate(dataset_range):
            if i == 0 or i == 3 or i == 6 or i == 9:
                image_list.append(dataset[n][0].asnumpy())
                
    distortion_types = ["saturation increase", "saturation decrease", "brightness increase",
                        "brightness decrease", "hue", "contrast increase", "contrast decrease", "noise"]
    distortion_levels = ["0", "3", "6", "9"]
    
    fig, axs = plt.subplots(8, 4, figsize=(12*side_ratio, 24))
    for i, ax in enumerate(axs.flat):
        ax.imshow(image_list[i])
        ax.axis("off")
        
        if i % 4 == 0:
            anchored_text = AnchoredText(distortion_types[i//4], loc=2, prop=dict(size=19))
            ax.add_artist(anchored_text)
            
        if i // 4 == 7:
            ax.set_title(distortion_levels[i % 4], y=-0.15, fontsize=17)
            
    plt.suptitle("Distortion Level", y=0.095, fontsize=21)
    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    plt.show()
    
    return fig