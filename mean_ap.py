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

with open('data/datasets/class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)
    
IMG_SIZE = 416
IOU_THRESHHOLD = 0.5
BKGD_THRESHHOLD = 0.1
NUM_IMAGES = 100

def show_coco_img(coco_instance):
    """
    Takes an instance of a COCO dataset as input and returns a plot
    of an image including bounding boxes and class annotations.
    """
    img, label = coco_instance[0].asnumpy(), coco_instance[1]
    
    bounding_boxes = label[:, :4]
    class_ids = label[:, 4:5]
    
    img_plot = plt.imshow(img)
    for i in range(len(bounding_boxes)):
        r_x_min = int(bounding_boxes[i][0])
        r_y_min = int(bounding_boxes[i][1])
        r_width = int(bounding_boxes[i][2] - bounding_boxes[i][0])
        r_height = int(bounding_boxes[i][3] - bounding_boxes[i][1])
        if i == 0:
            plt.gca().add_patch(Rectangle((r_x_min,r_y_min),r_width,r_height,
                            edgecolor='g',
                            facecolor='none',
                            lw=2, label="true"))
        else:
            plt.gca().add_patch(Rectangle((r_x_min,r_y_min),r_width,r_height,
                            edgecolor='g',
                            facecolor='none',
                            lw=2))
        plt.text(r_x_min, r_y_min-5,
                 class_names[int(class_ids[i][0])],
                 c='g',
                 fontweight='bold')
    plt.legend()
    plt.axis("off")
    plt.show();
    return img_plot

def preprocess_image(coco_image, img_size=IMG_SIZE):
    """
    Takes an image from a COCO dataset as input and preprocesses it for a YOLO prediction.
    """
    image = tf.constant(coco_image.asnumpy())
    image = tf.image.resize(image, size=[img_size, img_size]) / 255
    yolo_image = tf.expand_dims(image, 0)
    
    return yolo_image

def normalize_bounding_boxes(image, bounding_boxes):
    """
    Takes an image and list bounding boxes in absolute coordinates 
    [x1, y1, x2, y2]
    as input, normalizes bounding boxes coordinates and returns
    normalized bounding boxes.
    """
    img_width = image.shape[1]
    img_height = image.shape[0]
    norm_bounding_boxes = np.zeros(bounding_boxes.shape)
    for i in range(len(norm_bounding_boxes)):
        norm_bounding_boxes[i, 0] = bounding_boxes[i, 0] / img_width
        norm_bounding_boxes[i, 1] = bounding_boxes[i, 1] / img_height
        norm_bounding_boxes[i, 2] = bounding_boxes[i, 2] / img_width
        norm_bounding_boxes[i, 3] = bounding_boxes[i, 3] / img_height
    return norm_bounding_boxes

def convert_bounding_boxes(norm_bound_boxes):
    """
    Takes list of nomalized bounding boxes 
    [x1, y1, x2, y2] as input
    and converts them to YOLO output format (shape=(1, 100, 4))
    """
    converted_bound_boxes = np.zeros(shape=(1,100,4))
    for i in range(len(norm_bound_boxes)):
        converted_bound_boxes[0, i] = norm_bound_boxes[i]
    return converted_bound_boxes

def conv_coco_label_to_yolo_format(coco_image, coco_label):
    """
    Takes an image and its corresponding label in coco dataset
    format as input, converts label to YOLO output format and 
    returns YOLO - label.
    """
    bounding_boxes = coco_label[:, :4]
    class_ids = coco_label[:, 4:5]
    
    boxes = normalize_bounding_boxes(coco_image, bounding_boxes)
    boxes = convert_bounding_boxes(boxes)
    
    scores = np.zeros((1, 100))
    for i in range(len(bounding_boxes)):
        scores[0, i] = 1
    
    classes = np.zeros((1, 100), dtype=int)
    for i in range(len(class_ids)):
        classes[0, i] = int(class_ids[i, 0])
    
    nums = np.array(len(bounding_boxes), dtype=int).reshape((1,))
    
    return boxes, scores, classes, nums

def plot_bound_boxes(img, pred_label, true_label=None):
    """
    Takes image, predicted label and (optinal) true label in YOLO output format as input
    and returns an image with ploted bounding boxes.
    """
    pred_boxes = pred_label[0][0]
    pred_probas = pred_label[1][0]
    class_ids_pred = pred_label[2][0]
    if true_label:
        true_boxes = true_label[0][0]
        class_ids_true = true_label[2][0]
        
    img_width = img.shape[1]
    img_height = img.shape[0]
    
    boxed_img = plt.imshow(img)
        
    for i in range(pred_label[3][0]):
        r_x_min = int(pred_boxes[i][0] * img_width)
        r_y_min = int(pred_boxes[i][1] * img_height)
        r_width = int((pred_boxes[i][2] - pred_boxes[i][0]) * img_width)
        r_height = int((pred_boxes[i][3] - pred_boxes[i][1]) * img_height)
        if i == 0:
            plt.gca().add_patch(Rectangle((r_x_min,r_y_min),r_width,r_height,
                            edgecolor='red',
                            facecolor='none',
                            lw=2, label="pred"))
        else:
            plt.gca().add_patch(Rectangle((r_x_min,r_y_min),r_width,r_height,
                            edgecolor='red',
                            facecolor='none',
                            lw=2))
        plt.text(r_x_min, r_y_min-5,
                 class_names[int(class_ids_pred[i])] + f" {pred_probas[i]:.3f}",
                 c='r',
                 fontweight='bold')
    
    if true_label:
        for i in range(true_label[3][0]):
            r_x_min = int(true_boxes[i][0] * img_width)
            r_y_min = int(true_boxes[i][1] * img_height)
            r_width = int((true_boxes[i][2] - true_boxes[i][0]) * img_width)
            r_height = int((true_boxes[i][3] - true_boxes[i][1]) * img_height)
            if i == 0:
                plt.gca().add_patch(Rectangle((r_x_min,r_y_min),r_width,r_height,
                                edgecolor='green',
                                facecolor='none',
                                lw=2, label="true"))
            else:
                plt.gca().add_patch(Rectangle((r_x_min,r_y_min),r_width,r_height,
                                edgecolor='green',
                                facecolor='none',
                                lw=2))
            plt.text(r_x_min, r_y_min+r_height+11,
                     class_names[int(class_ids_true[i])],
                     c='g',
                     fontweight='bold')
    plt.axis("off")
    plt.legend()
    plt.show();
    return boxed_img

def get_iou(box_true, box_pred):
    """
    Takes normalized ground truth and predicted bounding boxes 
    in form [ x1, y1, x2, y2] as input and returns intersection over union (IoU).
    """
    # Coordinates of bounding boxes edge points
    x1_t, x2_t, y1_t, y2_t = box_true[0], box_true[2], box_true[1], box_true[3]
    x1_p, x2_p, y1_p, y2_p = box_pred[0], box_pred[2], box_pred[1], box_pred[3]
    # Width & height of bounding boxes
    w_t, w_p = x2_t - x1_t, x2_p - x1_p
    h_t, h_p = y2_t - y1_t, y2_p - y1_p
    # Surface area of bounding boxes
    s_t, s_p = w_t * h_t, w_p * h_p
    # Width of intersection
    if x1_p >= x2_t:
        w_i = 0
    elif x1_t <= x1_p < x2_t and x2_p > x2_t:
        w_i = x2_t - x1_p
    elif x1_t <= x1_p < x2_t and x1_t < x2_p <= x2_t:
        w_i = w_p
    elif x1_p < x1_t and x1_t < x2_p <= x2_t:
        w_i = x2_p - x1_t
    elif x1_p < x1_t and x2_p <= x1_t:
        w_i = 0
    else:
        w_i = w_t
    # Height of intersection    
    if y1_p >= y2_t:
        h_i = 0
    elif y1_t <= y1_p < y2_t and y2_p > y2_t:
        h_i = y2_t - y1_p
    elif y1_t <= y1_p < y2_t and y1_t < y2_p <= y2_t:
        h_i = h_p
    elif y1_p < y1_t and y1_t < y2_p <= y2_t:
        h_i = y2_p - y1_t
    elif y1_p < y1_t and y2_p <= y1_t:
        h_i = 0
    else:
        h_i = h_t
    # Surface area of intersection    
    s_i = w_i * h_i
    # Surface area of union
    s_u = s_t + s_p - s_i
    # Intersection over union
    iou = s_i/s_u
    return iou

def get_iou_list_for_one_prediction(true_label, pred_label):
    """
    Takes ground truth and predicted labels of a single image in YOLO output format as inputs
    and returns a list of IoU values for every predicted bounding box.
    """
    true_num, pred_num = true_label[3][0], pred_label[3][0]
    true_boxes, pred_boxes = true_label[0][0], pred_label[0][0]
    
    iou_list = []
    
    for i in range(pred_num):
        pred_box_to_check = pred_boxes[i]
        iou_list_1pred = []
        
        for j in range(true_num):
            iou_list_1pred.append(get_iou(pred_box_to_check, true_boxes[j]))
           
        iou_array_1pred = np.array(iou_list_1pred)
        iou = np.max(iou_array_1pred)
        iou_list.append(iou)
        
    return iou_list

def get_eval_df_for_one_prediction(true_label, pred_label, threshhold=IOU_THRESHHOLD, bkgd_thresh=BKGD_THRESHHOLD):
    """
    Takes ground truth and predicted labels of a single image in YOLO output format as inputs
    and returns a DataFrame of IoU values for every predicted bounding box.
    """
    true_num, pred_num = true_label[3][0], pred_label[3][0]
    true_boxes, pred_boxes = true_label[0][0], pred_label[0][0]
    true_class_ids, pred_class_ids = true_label[2][0], pred_label[2][0]
    pred_objectness = pred_label[1][0]
    
    eval_df = pd.DataFrame(columns=["objectness", "pred_class", "true_class", "pred_box", "true_box", "iou", "status", "errors"])
    
    true_index_used = []
    
    for i in range(pred_num):
        pred_box = pred_boxes[i]
        pred_class = class_names[pred_class_ids[i]]
        objectness = pred_objectness[i]
        errors = []
        
        iou_list_1pred = []

        for j in range(true_num):
            iou_list_1pred.append(get_iou(pred_box, true_boxes[j]))

        iou_array_1pred = np.array(iou_list_1pred)
        iou = np.max(iou_array_1pred)

        if iou > bkgd_thresh:
            true_index = np.argmax(iou_array_1pred)
            true_index_used.append(true_index)
            true_class = class_names[true_class_ids[true_index]]
            true_box = true_boxes[true_index]
            status = None
            
            if iou < threshhold or pred_class != true_class:
                status = "FP"
                if iou < threshhold:
                    errors.append("loc")
                if pred_class != true_class:
                    errors.append("cls")            
            
        else:
            true_class = None
            true_box = None
            status = "FP"
            errors.append("bkgd")
            true_index_used.append(None)

        eval_df.loc[i] = [objectness, pred_class, true_class, pred_box, true_box, iou, status, errors]   
    
    true_index_dict = {}
    for true_index in set(true_index_used):
        true_index_dict[true_index] = []
        for det_index in range(len(true_index_used)):
            if true_index_used[det_index] == true_index:
                true_index_dict[true_index].append(det_index)
                
    for true_index in true_index_dict.keys():
        if true_index == None:
            pass
        
        elif len(true_index_dict[true_index]) == 1:
            if eval_df.loc[true_index_dict[true_index][0], "iou"] >= threshhold and eval_df.loc[true_index_dict[true_index][0], "pred_class"] == eval_df.loc[true_index_dict[true_index][0], "true_class"]:
                eval_df.loc[true_index_dict[true_index][0], "status"] = "TP"
            else:
                pass
            
        elif len(true_index_dict[true_index]) > 1:
            ious_duplicate = []
            for det_index in true_index_dict[true_index]:
                ious_duplicate.append(eval_df["iou"].loc[det_index])
            for i in range(len(true_index_dict[true_index])):
                if i == ious_duplicate.index(max(ious_duplicate)):
                    if eval_df.loc[true_index_dict[true_index][i], "iou"] >= threshhold and eval_df.loc[true_index_dict[true_index][i], "pred_class"] == eval_df.loc[true_index_dict[true_index][i], "true_class"]:
                        eval_df.loc[true_index_dict[true_index][i], "status"] = "TP"
                    else:
                        pass
                else:
                    eval_df.loc[true_index_dict[true_index][i], "status"] = "FP"
                    eval_df.loc[true_index_dict[true_index][i], "errors"].append("duplicate")
    

    while None in true_index_used:
        true_index_used.remove(None)
    
    true_index_used = sorted(set(true_index_used), reverse=True)
    true_index_left = list(range(true_num))
    for index in true_index_used:
        del true_index_left[index]
    
    
    if len(true_index_left) > 0:
        
        count = pred_num
        
        for i in true_index_left:    
            objectness, pred_class, pred_box = None, None, None
            errors = ["missed"]
            iou = 0
            status = "FN"
            true_class = class_names[true_class_ids[i]]
            true_box = true_boxes[i]
            eval_df.loc[count] = [objectness, pred_class, true_class, pred_box, true_box, iou, status, errors]
            count += 1
                    
    return eval_df

def plot_eval_one_img(coco_instance, yolo):
    
    coco_image = coco_instance[0].asnumpy()
    coco_label = coco_instance[1]
    
    true_label = conv_coco_label_to_yolo_format(coco_image, coco_label)
    
    processed_image = preprocess_image(coco_instance[0])
    pred_label = yolo.predict(processed_image)
    
    eval_df = get_eval_df_for_one_prediction(true_label, pred_label)
    
    num_rows = math.ceil(len(eval_df) / 2)
    num_cols = 2
    
    img_width = coco_image.shape[1]
    img_height = coco_image.shape[0]
    
    eval_fig = plt.figure(figsize=(num_cols*4*img_width/img_height, num_rows*4))
    
    for i in range(len(eval_df)):
        row = eval_df.loc[i]
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(coco_image)
        
        if row["pred_class"]:
            pred_box = row["pred_box"]
            r_x_min = int(pred_box[0] * img_width)
            r_y_min = int(pred_box[1] * img_height)
            r_width = int((pred_box[2] - pred_box[0]) * img_width)
            r_height = int((pred_box[3] - pred_box[1]) * img_height)
            plt.gca().add_patch(Rectangle((r_x_min,r_y_min),r_width,r_height,
                            edgecolor='red',
                            facecolor='none',
                            lw=2, label="pred"))
            plt.text(r_x_min, r_y_min-4,
                     row["pred_class"] + f' {row["objectness"]:.3f}',
                     c='r',
                     fontweight='bold', va='bottom')
            
        if row["true_class"]:
            true_box = row["true_box"]
            r_x_min = int(true_box[0] * img_width)
            r_y_min = int(true_box[1] * img_height)
            r_width = int((true_box[2] - true_box[0]) * img_width)
            r_height = int((true_box[3] - true_box[1]) * img_height)
            plt.gca().add_patch(Rectangle((r_x_min,r_y_min),r_width,r_height,
                            edgecolor='g',
                            facecolor='none',
                            lw=2, label="true"))
            plt.text(r_x_min, r_y_min+r_height+4,
                     row["true_class"],
                     c='g',
                     fontweight='bold', va='top')    
        
        if len(row["errors"]) > 0:
            plt.text(7, 10,
                     "IoU: " + f'{row["iou"]:.2f}' + "\nStatus: " + row["status"] + "\nErrors: " + str(row["errors"]),
                     c="magenta", fontsize="large", fontweight="bold", va='top')
        else:
            plt.text(7, 10,
                     "IoU: " + f'{row["iou"]:.2f}' + "\nStatus: " + row["status"],
                     c="magenta", fontsize="large", fontweight="bold", va='top')
            
        plt.legend()
        
        plt.axis("off")
        
    plt.tight_layout()
    plt.show()
    return eval_fig

def get_big_df(val_dataset, yolo, num_images=NUM_IMAGES, n=0):
    
    if n + NUM_IMAGES <= len(val_dataset):
        
        big_df = pd.DataFrame(columns=["objectness", "pred_class", "true_class", "pred_box", "true_box", "iou", "status", "errors", "img"])

        for i in range(n, n+num_images):
            coco_image = val_dataset[i][0].asnumpy()
            coco_label = val_dataset[i][1]

            true_label = conv_coco_label_to_yolo_format(coco_image, coco_label)

            processed_image = preprocess_image(val_dataset[i][0])
            pred_label = yolo.predict(processed_image)

            eval_df = get_eval_df_for_one_prediction(true_label, pred_label)
            eval_df["img"] = i - n
            big_df = big_df.append(eval_df)

        big_df = big_df.reset_index(drop=True)

        return big_df
    
    else:
        raise ValueError("n + NUM_IMAGES > len(dataset)")
        
def get_class_dfs(big_df):
    class_dfs = []
    
    fn_df = big_df[big_df["status"]=="FN"]
    fclf_ls = []
    for i in range(len(big_df)):
        if "cls" in big_df.loc[i, "errors"]:
            fclf_ls.append(i)
    fclf_df = big_df.loc[fclf_ls]
    fn_fclf_df = fn_df.append(fclf_df)
    
    unique_classes = big_df["pred_class"].append(big_df["true_class"]).unique()
    if np.isin(None, unique_classes):
        unique_classes = np.delete(unique_classes, np.where(unique_classes==None))
    
    for cls in unique_classes:
        pred_cls_df = big_df[big_df["pred_class"]==cls]
        
        num_preds = len(pred_cls_df)
        
        fn_fclf_cls_df = fn_fclf_df[fn_fclf_df["true_class"]==cls]
        cls_df = pred_cls_df.append(fn_fclf_cls_df)
        
        unique_true_boxes = cls_df["true_box"].apply(lambda x: tuple(x) if type(x) is np.ndarray else x).unique()
        if np.isin(None, unique_true_boxes):
            unique_true_boxes = np.delete(unique_true_boxes, np.where(unique_true_boxes==None))
        num_trues = len(unique_true_boxes)
        
        cls_df = cls_df.sort_values("objectness", ascending=False)
        cls_df = cls_df.reset_index(drop=True)
        
        cls_df["precision"] = None
        cls_df["recall"] = None
        
        correct_pred_num = 0
        for i in range(num_preds):
            if cls_df.loc[i, "status"] == "TP":
                correct_pred_num += 1
            
            cls_df.loc[i, "precision"] = correct_pred_num / (i + 1)
            cls_df.loc[i, "recall"] = correct_pred_num / num_trues
            
        class_dfs.append(cls_df)
        
    return class_dfs

def exact_ap(class_df):
    
    if class_df.loc[0, "precision"] == None:
        pl_recall = pd.Series([0, 1])
        pl_precision = pd.Series([0, 0])
        
    else:    
        if np.any(class_df["precision"].isin([None])):
            index = class_df["precision"].isin([None]).idxmax()
            recall = class_df.loc[:index-1, "recall"]
            precision = class_df.loc[:index-1, "precision"]
        else:
            recall = class_df["recall"]
            precision = class_df["precision"]

        pl_recall = recall.append(pd.Series([recall.iloc[-1]])).reset_index(drop=True).copy()
        pl_precision = precision.append(pd.Series([0])).reset_index(drop=True).copy()
        
        if pl_recall.iloc[-1] != 1:
            pl_recall = pl_recall.append(pd.Series([1])).reset_index(drop=True)
            pl_precision = pl_precision.append(pd.Series([0])).reset_index(drop=True)

        if pl_precision.loc[0] == 1:
            pl_recall = pd.concat([pd.Series([0]), pl_recall]).reset_index(drop=True)
            pl_precision = pd.concat([pd.Series([1]), pl_precision]).reset_index(drop=True)
                    
    exact_ap = auc(pl_recall, pl_precision)
 
    return exact_ap

def smoothed_ap(class_df):
    
    if class_df.loc[0, "precision"] == None:
        pl_recall = pd.Series([0, 1])
        smooth_precision = pd.Series([0, 0])
        
    else:    
        if np.any(class_df["precision"].isin([None])):
            index = class_df["precision"].isin([None]).idxmax()
            recall = class_df.loc[:index-1, "recall"]
            precision = class_df.loc[:index-1, "precision"]
        else:
            recall = class_df["recall"]
            precision = class_df["precision"]

        pl_recall = recall.append(pd.Series([recall.iloc[-1]])).reset_index(drop=True).copy()
        pl_precision = precision.append(pd.Series([0])).reset_index(drop=True).copy()
        
        if pl_recall.iloc[-1] != 1:
            pl_recall = pl_recall.append(pd.Series([1])).reset_index(drop=True)
            pl_precision = pl_precision.append(pd.Series([0])).reset_index(drop=True)

        if pl_precision.loc[0] == 1:
            pl_recall = pd.concat([pd.Series([0]), pl_recall]).reset_index(drop=True)
            pl_precision = pd.concat([pd.Series([1]), pl_precision]).reset_index(drop=True)
                    
        smooth_precision = pd.Series([], name="smooth_precision", dtype="float32")
        for i in range(len(pl_precision)):
            smooth_precision.loc[i] = pl_precision.loc[i:].max()
    
    smoothed_ap = auc(pl_recall, smooth_precision)
 
    return smoothed_ap

def interpolated_ap(class_df):
    
    if class_df.loc[0, "precision"] == None:
        interpolated_ap = 0
        
    else:    
        if np.any(class_df["precision"].isin([None])):
            index = class_df["precision"].isin([None]).idxmax()
            recall = class_df.loc[:index-1, "recall"]
            precision = class_df.loc[:index-1, "precision"]
        else:
            recall = class_df["recall"]
            precision = class_df["precision"]

        pl_recall = recall.append(pd.Series([recall.iloc[-1]])).reset_index(drop=True).copy()
        pl_precision = precision.append(pd.Series([0])).reset_index(drop=True).copy()
        
        if pl_recall.iloc[-1] != 1:
            pl_recall = pl_recall.append(pd.Series([1])).reset_index(drop=True)
            pl_precision = pl_precision.append(pd.Series([0])).reset_index(drop=True)

        if pl_precision.loc[0] == 1:
            pl_recall = pd.concat([pd.Series([0]), pl_recall]).reset_index(drop=True)
            pl_precision = pd.concat([pd.Series([1]), pl_precision]).reset_index(drop=True)
        
        interp_recall = pd.Series(np.linspace(0, 1.0, 101), name="interp_recall")
        interp_precision = pd.Series([], dtype="float32", name="interp_precision")
        for i in range(len(interp_recall)):
            interp_precision.loc[i] = pl_precision[pl_recall >= interp_recall[i]].max()

        interpolated_ap = interp_precision.mean()
 
    return interpolated_ap

def calculate_exact_map(class_dfs):
    ap_arr = np.array([])
    for i in range(len(class_dfs)):
        ap_arr = np.append(ap_arr, exact_ap(class_dfs[i]))
        
    return ap_arr.mean()

def calculate_smoothed_map(class_dfs):
    ap_arr = np.array([])
    for i in range(len(class_dfs)):
        ap_arr = np.append(ap_arr, smoothed_ap(class_dfs[i]))
        
    return ap_arr.mean()

def calculate_interpolated_map(class_dfs):
    ap_arr = np.array([])
    for i in range(len(class_dfs)):
        ap_arr = np.append(ap_arr, interpolated_ap(class_dfs[i]))
        
    return ap_arr.mean()

def plot_prec_rec_curve(class_df, show_values=True, show_smoothed=False, show_interpolation=False, show_plot=True):  
    
    if class_df.loc[0, "precision"] == None:
        pl_recall = pd.Series([0, 1])
        pl_precision = pd.Series([0, 0])
        plt.plot(pl_recall, pl_precision, color="blue", linewidth=2)
        str_exact_ap = f'Exact AP: 0.00'
        str_smoothed_ap = f'Smoothed AP: 0.00'
        str_interpolated_ap = f'Interpolated AP: 0.00'
        
    else:    
        if np.any(class_df["precision"].isin([None])):
            index = class_df["precision"].isin([None]).idxmax()
            recall = class_df.loc[:index-1, "recall"]
            precision = class_df.loc[:index-1, "precision"]
        else:
            recall = class_df["recall"]
            precision = class_df["precision"]

        pl_recall = recall.append(pd.Series([recall.iloc[-1]])).reset_index(drop=True).copy()
        pl_precision = precision.append(pd.Series([0])).reset_index(drop=True).copy()
        
        if pl_recall.iloc[-1] != 1:
            pl_recall = pl_recall.append(pd.Series([1])).reset_index(drop=True)
            pl_precision = pl_precision.append(pd.Series([0])).reset_index(drop=True)

        if pl_precision.loc[0] == 1:
            pl_recall = pd.concat([pd.Series([0]), pl_recall]).reset_index(drop=True)
            pl_precision = pd.concat([pd.Series([1]), pl_precision]).reset_index(drop=True)
                    
        plt.plot(pl_recall, pl_precision, color="blue", linewidth=2, zorder=1, label="exact")
        exact_ap = auc(pl_recall, pl_precision)
        str_exact_ap = f'Exact AP: {exact_ap:.2f}'
        
        if show_values:
            plt.scatter(recall, precision, c="limegreen", s=50, zorder=2, label="values")
        
    if show_smoothed:
        smooth_precision = pd.Series([], name="smooth_precision", dtype="float32")
        for i in range(len(pl_precision)):
            smooth_precision.loc[i] = pl_precision.loc[i:].max()
        
        plt.plot(pl_recall, smooth_precision, color="darkorange", linewidth=2, zorder=3, label="smoothed")
        smoothed_ap = auc(pl_recall, smooth_precision)
        str_smoothed_ap = f'Smoothed AP: {smoothed_ap:.2f}'
        
    if show_interpolation:
        interp_recall = pd.Series(np.linspace(0, 1.0, 101), name="interp_recall")
        interp_precision = pd.Series([], dtype="float32", name="interp_precision")
        for i in range(len(interp_recall)):
            interp_precision.loc[i] = pl_precision[pl_recall >= interp_recall[i]].max()
            
        plt.scatter(interp_recall, interp_precision, c="darkred", s=6, zorder=4, label="interpolated")
        interpolated_ap = interp_precision.mean()
        str_interpolated_ap = f'Interpolated AP: {interpolated_ap:.2f}'
        
    plt.xlim((-0.1, 1.1))
    plt.ylim((-0.1, 1.1))
    plt.axhline(y=0, color="black", linewidth=1)
    plt.axvline(x=0, color="black", linewidth=1)
    plt.xlabel("Recall", fontsize=13, fontweight="bold")
    plt.ylabel("Precision", fontsize=13, fontweight="bold")
    if class_df.loc[0, "pred_class"]:
        title_string = class_df.loc[0, "pred_class"][0].upper() + class_df.loc[0, "pred_class"][1:] + " class"
    else:
        title_string = class_df.loc[0, "true_class"][0].upper() + class_df.loc[0, "true_class"][1:] + " class"
    plt.title(title_string, fontsize=16, fontweight="bold")
    
    legend = plt.legend(fontsize="medium", frameon=True, fancybox=True, loc="best");
    
    # get the bounding box of the legend in display coordinates
    renderer = plt.gcf().canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer=renderer)

    # convert the display coordinates to data coordinates
    trans_data = plt.gca().transData.inverted()
    legend_box_data = trans_data.transform(legend_box)
    
    if show_smoothed and show_interpolation:
        str_ap = str_exact_ap + "\n" + str_smoothed_ap + "\n" + str_interpolated_ap
    elif show_smoothed:
        str_ap = str_exact_ap + "\n" + str_smoothed_ap
    elif show_interpolation:
        str_ap = str_exact_ap + "\n" + str_interpolated_ap
    else:
        str_ap = str_exact_ap
        
    plt.text(legend_box_data[0, 0]-0.06, legend_box_data[0, 1]-0.02, str_ap, fontsize=11, va="top")
    plt.grid(True)
    
    if show_plot:
        plt.show();
        
def plot_all_prec_rec_curves(class_dfs, show_smoothed=False, show_interpolation=False):
    num_cols = 3
    num_rows = math.ceil(len(class_dfs) / num_cols)
    
    fig = plt.figure(figsize=(7*num_cols, 5*num_rows))
    for i in range(len(class_dfs)):
        plt.subplot(num_rows, num_cols, i+1)
        if show_smoothed and show_interpolation:
            plot_prec_rec_curve(class_dfs[i], show_plot=False, show_smoothed=True, show_interpolation=True)
        elif show_smoothed:
            plot_prec_rec_curve(class_dfs[i], show_plot=False, show_smoothed=True)
        elif show_interpolation:
            plot_prec_rec_curve(class_dfs[i], show_plot=False, show_interpolation=True)
        else:
            plot_prec_rec_curve(class_dfs[i], show_plot=False)
    plt.tight_layout()
    plt.show();