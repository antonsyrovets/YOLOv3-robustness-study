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

from mean_ap import *
from ds_ranges import *

lst = ["saturation", "brightness", "hue", "contrast", "noise"]
permutations = list(itertools.permutations(lst))

def calculate_map_range(dataset_range, yolo, interpolated=True, exact=False, smoothed=False):
    map_range = []
    for dataset in dataset_range:
        big_df = get_big_df(dataset, yolo)
        class_dfs = get_class_dfs(big_df)
        if interpolated and not exact and not smoothed:
            mean_ap = calculate_interpolated_map(class_dfs)
        elif exact and not interpolated and not smoothed:
            mean_ap = calculate_exact_map(class_dfs)
        elif smoothed and not interpolated and not exact:
            mean_ap = calculate_smoothed_map(class_dfs)
        else:
            raise ValueError("Only one optional input parameter can be set to `True`.")
            
        map_range.append(mean_ap)
        
    return map_range

def create_image_range(dataset_range, n=0):
    image_range = []
    for dataset in dataset_range:
        image_range.append(dataset[n][0].asnumpy())
    return image_range

def add_image(ax, img, x_loc, y_loc, img_zoom=0.085):
    image = OffsetImage(img, zoom=img_zoom)
    image.image.axes = ax
    
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ab = AnnotationBbox(image, (x_loc, y_loc), xycoords=trans, boxcoords=("data", "axes fraction"),
                        box_alignment=(.5, 1), bboxprops=dict(edgecolor='none', facecolor='none', alpha=0), pad=0)
    ax.add_artist(ab)
    
def plot_map_range_saturation(map_range_saturation, dataset_range_saturation):
    
    saturation_range = np.linspace(-100, 100, len(map_range_saturation))
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rc("text", usetex=True)
    ax.plot(saturation_range, map_range_saturation, color="blue", marker="p", markersize=7)
    ax.set_title("YOLO Performance vs. Saturation", fontweight="bold", fontsize=16)
    ax.set_xlabel("Saturation Increment (HSV)", fontweight="bold", fontsize=14, labelpad=40)
    ax.set_ylabel("Mean $AP_{IoU=0.5}$", fontweight="bold", fontsize=14)
    ax.grid(True)
    ax.set_xticks(saturation_range)
    formatter = FuncFormatter(lambda x, _: f'+{x:.0f}' if x>0 else f'{x:.0f}')
    ax.xaxis.set_major_formatter(formatter)
    saturation_image_range = create_image_range(dataset_range_saturation, n=5)
    for i, image in enumerate(saturation_image_range):
        add_image(ax, image, saturation_range[i], -0.05)
    
    for i in range(len(map_range_saturation)):
        text = ax.text(saturation_range[i], map_range_saturation[i]+0.003, f"{map_range_saturation[i]:.2f}",
                       ha="center", va="center")
        
    return fig

def plot_map_range_saturation_up(map_range_saturation_up, dataset_range_saturation_up):
    
    saturation_range = np.linspace(0.0, 100.0, len(map_range_saturation_up))
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rc("text", usetex=True)
    ax.plot(saturation_range, map_range_saturation_up, color="darkred", marker="p", markersize=7)
    ax.set_title("YOLO Performance vs. Saturation", fontweight="bold", fontsize=16)
    ax.set_xlabel("Saturation Increment (HSV)", fontweight="bold", fontsize=14, labelpad=37)
    ax.set_ylabel("Mean $AP_{IoU=0.5}$", fontweight="bold", fontsize=14)
    ax.grid(True)
    ax.set_xticks(saturation_range)
    formatter = FuncFormatter(lambda x, _: f'+{x:.0f}' if x>0 else f'{x:.0f}')
    ax.xaxis.set_major_formatter(formatter)
    saturation_up_image_range = create_image_range(dataset_range_saturation_up, n=5)
    for i, image in enumerate(saturation_up_image_range):
        add_image(ax, image, saturation_range[i], -0.05, img_zoom=0.076)
    
    for i in range(len(map_range_saturation_up)):
        text = ax.text(saturation_range[i]+1.5, map_range_saturation_up[i]+0.003, f"{map_range_saturation_up[i]:.2f}",
                       ha="center", va="center")
        
    return fig

def plot_map_range_saturation_down(map_range_saturation_down, dataset_range_saturation_down):
    
    saturation_range = np.linspace(0.0, -100.0, len(map_range_saturation_down))
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rc("text", usetex=True)
    ax.plot(saturation_range, map_range_saturation_down, color="mediumblue", marker="p", markersize=7)
    ax.set_title("YOLO Performance vs. Saturation", fontweight="bold", fontsize=16)
    ax.set_xlabel("Saturation Increment (HSV)", fontweight="bold", fontsize=14, labelpad=37)
    ax.set_ylabel("Mean $AP_{IoU=0.5}$", fontweight="bold", fontsize=14)
    ax.grid(True)
    ax.set_xticks(saturation_range)
    formatter = FuncFormatter(lambda x, _: f'{x:.0f}')
    ax.xaxis.set_major_formatter(formatter)
    ax.invert_xaxis()
    saturation_down_image_range = create_image_range(dataset_range_saturation_down, n=5)
    for i, image in enumerate(saturation_down_image_range):
        add_image(ax, image, saturation_range[i], -0.05, img_zoom=0.076)
    
    for i in range(len(map_range_saturation_down)):
        text = ax.text(saturation_range[i]-0.7, map_range_saturation_down[i]+0.002, f"{map_range_saturation_down[i]:.2f}",
                       ha="center", va="center")
        
    return fig

def plot_map_range_saturation_up_down(map_range_saturation_up, map_range_saturation_down,
                                      dataset_range_saturation_up, dataset_range_saturation_down):
    
    saturation_range = np.linspace(0.0, 100.0, len(map_range_saturation_down))
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rc("text", usetex=True)
    
    ax.plot(saturation_range, map_range_saturation_up, color="darkred", marker="p", markersize=7)
    ax.set_title("YOLO Performance vs. Saturation", fontsize=16)
    ax.set_ylabel("Mean $AP_{IoU=0.5}$", fontsize=14)
    ax.grid(True)
    for i in range(len(map_range_saturation_up)):
        text = ax.text(saturation_range[i]-1.5, map_range_saturation_up[i]-0.004,
                       f"{map_range_saturation_up[i]:.2f}",
                       ha="center", va="center", color="darkred")
        
    ax.set_xticks(saturation_range)
    ax.tick_params(axis='x', labelcolor='darkred')
    formatter = FuncFormatter(lambda x, _: f'+{x:.0f}' if x > 0 else f'{x:.0f}')
    ax.xaxis.set_major_formatter(formatter)
    saturation_up_image_range = create_image_range(dataset_range_saturation_up, n=5)
    for i, image in enumerate(saturation_up_image_range):
        add_image(ax, image, saturation_range[i], -0.05, img_zoom=0.077)
    
    saturation_range_down = np.linspace(0.0, -100.0, len(map_range_saturation_down))
    ax2 = ax.twiny()
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_position(('outward', 56))
    ax2.set_xticks(saturation_range_down)
    ax2.xaxis.set_major_formatter(formatter)
    ax2.set_xlabel("Saturation Increment (HSV)", fontweight="bold", fontsize=14, labelpad=38)
    ax2.plot(saturation_range_down, map_range_saturation_down, color="mediumblue", marker="p", markersize=7)
    ax2.invert_xaxis()
    ax2.tick_params(axis='x', labelcolor='mediumblue')
    
    for i in range(len(map_range_saturation_down)):
        if i != 1:
            text = ax2.text(saturation_range_down[i]-0.8, map_range_saturation_down[i]+0.004,
                            f"{map_range_saturation_down[i]:.2f}",
                            ha="center", va="center", color="mediumblue")
        else:
            text = ax2.text(saturation_range_down[i]+1.5, map_range_saturation_down[i]-0.004,
                            f"{map_range_saturation_down[i]:.2f}",
                            ha="center", va="center", color="mediumblue")
            
    saturation_down_image_range = create_image_range(dataset_range_saturation_down, n=5)
    for i, image in enumerate(saturation_down_image_range):
        add_image(ax, image, saturation_range[i], -0.218, img_zoom=0.077)
            
    return fig

def plot_map_range_brightness_up_down(map_range_brightness_up, map_range_brightness_down,
                                      dataset_range_brightness_up, dataset_range_brightness_down):
    
    brightness_range = np.linspace(0.0, 100.0, len(map_range_brightness_down))
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rc("text", usetex=True)
    
    ax.plot(brightness_range, map_range_brightness_up, color="darkviolet", marker="p", markersize=7)
    ax.set_title("YOLO Performance vs. Brightness", fontweight="bold", fontsize=16)
    ax.set_ylabel("Mean $AP_{IoU=0.5}$", fontweight="bold", fontsize=14)
    ax.grid(True)
    for i in range(len(map_range_brightness_up)):
        text = ax.text(brightness_range[i]+0.7, map_range_brightness_up[i]+0.015,
                       f"{map_range_brightness_up[i]:.2f}",
                       ha="center", va="center", color="darkviolet")
        
    ax.set_xticks(brightness_range)
    ax.tick_params(axis='x', labelcolor='darkviolet')
    formatter = FuncFormatter(lambda x, _: f'+{x:.0f}' if x > 0 else f'{x:.0f}')
    ax.xaxis.set_major_formatter(formatter)
    brightness_up_image_range = create_image_range(dataset_range_brightness_up, n=9)
    for i, image in enumerate(brightness_up_image_range):
        add_image(ax, image, brightness_range[i], -0.05, img_zoom=0.077)
    
    brightness_range_down = np.linspace(0.0, -100.0, len(map_range_brightness_down))
    ax2 = ax.twiny()
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_position(('outward', 59))
    ax2.set_xticks(brightness_range_down)
    ax2.xaxis.set_major_formatter(formatter)
    ax2.set_xlabel("Brightness Increment (HSV)", fontweight="bold", fontsize=14, labelpad=43)
    ax2.plot(brightness_range_down, map_range_brightness_down, color="darkgreen", marker="p", markersize=7)
    ax2.invert_xaxis()
    ax2.tick_params(axis='x', labelcolor='darkgreen')
    
    for i in range(len(map_range_brightness_down)):
        text = ax2.text(brightness_range_down[i]+1.9, map_range_brightness_down[i]-0.015,
                        f"{map_range_brightness_down[i]:.2f}",
                        ha="center", va="center", color="darkgreen")

    brightness_down_image_range = create_image_range(dataset_range_brightness_down, n=9)
    for i, image in enumerate(brightness_down_image_range):
        add_image(ax, image, brightness_range[i], -0.227, img_zoom=0.077)
            
    return fig

def plot_map_range_hue(map_range_hue, dataset_range_hue):
    
    hue_range = np.linspace(0, 360, len(map_range_hue))
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rc("text", usetex=True)
    ax.plot(hue_range, map_range_hue, color="indigo", marker="p", markersize=7)
    ax.set_title("YOLO Performace vs. Hue", fontweight="bold", fontsize=16)
    ax.set_xlabel("Hue Rotation", fontweight="bold", fontsize=14, labelpad=50)
    ax.set_ylabel("Mean $AP_{IoU=0.5}$", fontweight="bold", fontsize=14)
    ax.grid(True)
    ax.set_xticks(hue_range)
    formatter = FuncFormatter(lambda x, _: f'+{x:.0f}°' if x>0 else f'{x:.0f}°')
    ax.xaxis.set_major_formatter(formatter)
    hue_image_range = create_image_range(dataset_range_hue, n=7)
    for i, image in enumerate(hue_image_range):
        add_image(ax, image, hue_range[i], -0.05, img_zoom=0.07)
    
    for i in range(len(map_range_hue)):
        text = ax.text(hue_range[i]+11, map_range_hue[i]+0.0015, f"{map_range_hue[i]:.2f}",
                       ha="center", va="center")
        
    return fig

def plot_map_range_contrast_up_down(map_range_contrast_up, map_range_contrast_down,
                                    dataset_range_contrast_up, dataset_range_contrast_down):
    
    contrast_range = np.logspace(math.log(1.0, 10), math.log(50.0, 10), len(map_range_contrast_down))
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rc("text", usetex=True)
    
    ax.plot(contrast_range, map_range_contrast_up, color="m", marker="p", markersize=7)
    ax.set_xscale('log')
    ax.set_title("YOLO Performance vs. Contrast", fontweight="bold", fontsize=16)
    ax.set_ylabel("Mean $AP_{IoU=0.5}$", fontweight="bold", fontsize=14)
    ax.grid(True)
    for i in range(len(map_range_contrast_up)):
        text = ax.text(contrast_range[i], map_range_contrast_up[i]+0.015,
                       f"{map_range_contrast_up[i]:.2f}",
                       ha="center", va="center", color="m")
        
    ax.set_xticks(contrast_range)
    ax.set_xticks([], minor=True)
    ax.tick_params(axis='x', labelcolor='m')
    formatter = FuncFormatter(lambda x, _:f'{x:.2f}')
    ax.xaxis.set_major_formatter(formatter)
    contrast_up_image_range = create_image_range(dataset_range_contrast_up, n=45)
    for i, image in enumerate(contrast_up_image_range):
        add_image(ax, image, contrast_range[i], -0.05)
    
    contrast_range_down = np.linspace(1.0, 0.0, len(map_range_contrast_down))
    ax2 = ax.twiny()
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_position(('outward', 70))
    ax2.set_xticks(contrast_range_down)
    ax2.xaxis.set_major_formatter(formatter)
    ax2.set_xlabel("`contrast_factor` Parameter of `tf.image.adjust_contrast()`", fontweight="bold", fontsize=14, labelpad=60)
    ax2.plot(contrast_range_down, map_range_contrast_down, color="c", marker="p", markersize=7)
    ax2.invert_xaxis()
    ax2.tick_params(axis='x', labelcolor='c')
    
    
    for i in range(len(map_range_contrast_down)):
        text = ax2.text(contrast_range_down[i]+0.02, map_range_contrast_down[i]-0.015,
                        f"{map_range_contrast_down[i]:.2f}",
                        ha="center", va="center", color="c")

    contrast_down_image_range = create_image_range(dataset_range_contrast_down, n=45)
    for i, image in enumerate(contrast_down_image_range):
        add_image(ax, image, contrast_range[i], -0.262)
            
    return fig

def plot_map_range_noise(map_range_noise, dataset_range_noise):
    
    noise_range = tf.linspace(1, len(map_range_noise), len(map_range_noise))
    
    noise_level = tf.linspace(0.0, 0.3, len(map_range_noise))
    std = tf.linspace(0.0, 2.0, len(map_range_noise))
    salt_pepper_ratio = tf.linspace(0.0, 0.1, len(map_range_noise))
    xticklabels = []
    for i in range(len(map_range_noise)):
        xtick_label = f'{noise_level[i]:.2f}\n{std[i]:.2f}\n{salt_pepper_ratio[i]:.2f}'
        xticklabels.append(xtick_label)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rc("text", usetex=True)
    ax.plot(noise_range, map_range_noise, color="darkolivegreen", marker="p", markersize=7, linewidth=2)
    ax.set_title("YOLO Performance vs. Noise", fontweight="bold", fontsize=16)
    ax.set_xlabel("noise_level\nstd_dev\nsalt_pepper_ratio ", fontweight="bold", fontsize=14, labelpad=55)
    ax.set_ylabel("Mean $AP_{IoU=0.5}$", fontweight="bold", fontsize=14)
    ax.grid(True)
    ax.set_xticks(noise_range)
    ax.set_xticklabels(xticklabels)
    noise_image_range = create_image_range(dataset_range_noise, n=1)
    for i, image in enumerate(noise_image_range):
        add_image(ax, image, noise_range[i], -0.115, img_zoom=0.077)
    
    for i in range(len(map_range_noise)):
        text = ax.text(noise_range[i]+0.15, map_range_noise[i]+0.015, f"{map_range_noise[i]:.2f}",
                       ha="center", va="center")
        
    return fig

def plot_map_distortions_comparisson(map_range_saturation, map_range_brightness, map_range_hue,
                                     map_range_contrast, map_range_noise, increase=True):
    
    distortion_levels = tf.linspace(0, 10, 11)
#    hue_rotation = tf.linspace(0, 10, 12)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rc("text", usetex=True)
    if increase:
        ax.plot(distortion_levels, map_range_saturation, color="darkred", marker="p", markersize=7, linewidth=2, label="saturation")
        ax.plot(distortion_levels, map_range_brightness, color="darkviolet", marker="p", markersize=7, linewidth=2, label="brightness")
        ax.plot(distortion_levels, map_range_hue, color="indigo", marker="p", markersize=7, linewidth=2, label="hue")
        ax.plot(distortion_levels, map_range_contrast, color="m", marker="p", markersize=7, linewidth=2, label="contrast")
        ax.plot(distortion_levels, map_range_noise, color="darkolivegreen", marker="p", markersize=7, linewidth=2, label="noise")
        ax.set_title("YOLO Performance vs. Photometric Transformations (Increase)", fontsize=16)
    else:
        ax.plot(distortion_levels, map_range_saturation, color="mediumblue", marker="p", markersize=7, linewidth=2, label="saturation")
        ax.plot(distortion_levels, map_range_brightness, color="darkgreen", marker="p", markersize=7, linewidth=2, label="brightness")
        ax.plot(distortion_levels, map_range_hue, color="indigo", marker="p", markersize=7, linewidth=2, label="hue")
        ax.plot(distortion_levels, map_range_contrast, color="c", marker="p", markersize=7, linewidth=2, label="contrast")
        ax.plot(distortion_levels, map_range_noise, color="darkolivegreen", marker="p", markersize=7, linewidth=2, label="noise")
        ax.set_title("YOLO Performance vs. Photometric Transformations (Decrease)", fontsize=16)
        
    ax.set_xlabel("Distortion Level", fontsize=14)
    ax.set_ylabel("Mean $AP_{IoU=0.5}$", fontsize=14)
    ax.grid(True)
    ax.set_xticks(distortion_levels)
    ax.legend(fontsize="large")
    
#     for i in range(len(map_range_noise)):
#         text = ax.text(distortion_levels[i]+0.0, map_range_saturation[i]+0.015, f"{map_range_saturation[i]:.2f}",
#                        ha="center", va="center")
    
#     for i in range(len(map_range_noise)):
#         text = ax.text(distortion_levels[i]+0.0, map_range_brightness[i]+0.015, f"{map_range_brightness[i]:.2f}",
#                        ha="center", va="center")
    
#     for i in range(len(map_range_hue)):
#         text = ax.text(hue_rotation[i]+0.0, map_range_hue[i]+0.015, f"{map_range_hue[i]:.2f}",
#                        ha="center", va="center")
    
#     for i in range(len(map_range_noise)):
#         text = ax.text(distortion_levels[i]+0.0, map_range_contrast[i]+0.015, f"{map_range_contrast[i]:.2f}",
#                        ha="center", va="center")
    
#     for i in range(len(map_range_noise)):
#         if != 1:
#             text = ax.text(distortion_levels[i]+0.0, map_range_noise[i]+0.015, f"{map_range_noise[i]:.2f}",
#                            color="darkolivegreen", ha="center", va="center")
        
    return fig

def plot_map_range_permutations(permutations_map_range, distortion_level):
    index = np.arange(0, len(permutations_map_range))
    fig = plt.figure(figsize=(10, 6))
    plt.bar(index, permutations_map_range)
    plt.xlabel("Permutation Index", fontweight="bold", fontsize=14)
    plt.rc("text", usetex=True)
    plt.ylabel("Mean $AP_{IoU = 0.5}$", fontweight="bold", fontsize=14)
    plt.title("YOLO Performance vs. Distortion Order Permutations\nDistortion Level " + str(distortion_level),
              fontweight="bold", fontsize=16)
    minimum, maximum = permutations_map_range.min(), permutations_map_range.max()
    mean  = permutations_map_range.mean()
    median = np.median(permutations_map_range)
    std = permutations_map_range.std()
    text = plt.text(0.43, 0.4, f"min:{minimum:>7.2f}\nmax:{maximum:>7.2f}\nmean:{mean:>7.2f}\nmedian:{median:>7.2f}\nstd:{std:>7.2f}",
                    fontsize=14, fontname="Courier", transform=plt.gca().transAxes)
    text.set_bbox(dict(facecolor="white", edgecolor="black", boxstyle="round, pad=0.2", alpha=0.9))
    return fig

def plot_map_range_permutations_comparisson(permutations_1_map_range, permutations_2_map_range,
                                            permutations_3_map_range):
    index = np.arange(0, len(permutations))
    fig = plt.figure(figsize=(10, 6))
    plt.bar(index, permutations_1_map_range, label="distortion level 1", edgecolor="black", linewidth=0.5)
    plt.bar(index, permutations_2_map_range, label="distortion level 2", edgecolor="black", linewidth=0.5,
            color="springgreen")
    plt.bar(index, permutations_3_map_range, label="distortion level 3", edgecolor="black", linewidth=0.5,
            color="magenta")
    plt.xlabel("Permutation Index", fontweight="bold", fontsize=14)
    plt.rc("text", usetex=True)
    plt.ylabel("Mean $AP_{IoU = 0.5}$", fontweight="bold", fontsize=14)
    plt.title("YOLO Performance vs. Distortion Order Permutations",
              fontweight="bold", fontsize=16)
    plt.legend(fontsize="large")
    plt.ylim((0.0, 0.45))

    text = plt.text(0.1, 0.2, f"Common top:\ \ \ \ \ \ \ [57]\ \ = ('hue', 'brightness', 'contrast', 'noise', 'saturation')\n"
                              f"Common bottom: [109] = ('noise', 'hue', 'saturation', 'contrast', 'brightness')",
                    fontsize=14, fontname="Courier New", transform=plt.gca().transAxes)
    text.set_bbox(dict(facecolor="white", edgecolor="black", boxstyle="round, pad=0.2", alpha=0.9))
    return fig

def plot_permutations_top_bottom(dataset_range_permutations_1, dataset_range_permutations_2,
                                 dataset_range_permutations_3,
                                 top_bottom_index_list,  n=0):

    text_list = ["index " + str(top_bottom_index_list[0]) + " = common top",
                 "index " + str(top_bottom_index_list[1]) + " = common bottom"]
    nrows = len(top_bottom_index_list)
    
    img_shape = dataset_range_permutations_1[0][n][0].asnumpy().shape
    side_ratio = img_shape[1] / img_shape[0]
    
    fig, axs = plt.subplots(nrows, 3, figsize=(5*side_ratio*3, 5.35*nrows))
    for i, ax in enumerate(axs.flat):
        
        if i % 3 == 0:
            ax.imshow(dataset_range_permutations_1[top_bottom_index_list[i//3]][n][0].asnumpy())
            ax.axis("off")
            anchored_text = AnchoredText(text_list[i//3], loc=2, prop=dict(size=17))
            ax.add_artist(anchored_text)
            ax.set_title("1", y=-0.09, fontsize=19)
            
        if i % 3 == 1:
            ax.imshow(dataset_range_permutations_2[top_bottom_index_list[i//3]][n][0].asnumpy())
            ax.axis("off")
            ax.set_title("2", y=-0.09, fontsize=19)
            
        if i % 3 == 2:
            ax.imshow(dataset_range_permutations_3[top_bottom_index_list[i//3]][n][0].asnumpy())
            ax.axis("off")
            ax.set_title("3", y=-0.09, fontsize=19)
        
    plt.suptitle("Distortion Level", y=0.085, fontsize=21)
    plt.tight_layout
    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    plt.show()
    
    return fig