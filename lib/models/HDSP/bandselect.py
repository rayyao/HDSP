
import cv2
import numpy as np
from PIL import Image
import math
import random
import torch
import numpy as np
import os
import sys
import time
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torch.optim as optim
import os
import json
import numpy as np

order=None
def gen_config(args, videoname):

    if args.seq != '':
        # generate config from a sequence name

        # seq_home = 'datasets/OTB'
        seq_home = args.seq
        result_home = args.savepath

        seq_name = videoname
        img_dir = os.path.join(seq_home, seq_name, 'HSI')
        #gt_path = os.path.join(seq_home, seq_name,'groundtruth_rect.txt')

        img_list = os.listdir(img_dir)
        img_list.sort()
        img_list = [os.path.join(img_dir, x) for x in img_list]


        result_dir = result_home # os.path.join(result_home, seq_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

    return img_list

def X2Cube(img):

    B = [4, 4]
    skip = [4, 4]
    # Parameters
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1

    # Get Starting block indices
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])

    # Generate Depth indeces
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    img = out.reshape(M//4, N//4, 16)
    img = img / img.max() * 255 
    img.astype('uint8')
    return img


def Select(hsi_img):
    w=[]
    for kk in hsi_img:
     a=kk.reshape(1,-1)
     sum1=0
     l=np.sum(a)
     k=a.size
     pj=l/k
     for m in range(a.size):
         b=(a[0,m]-pj)**2
         sum1=sum1+b
         d=(sum1/(a.size))**0.5
     #d=(d,KK)
     w.append(d)
    w1=np.array(w)
    w1=np.argsort(-w1)
    orderW = w1[0:15]
    return orderW


def HSI2RGB(sample,order):
    ordersample0 = sample[order[0]]
    ordersample1 = sample[order[1]]
    ordersample2= sample[order[2]]
    com1 = np.array([ordersample0,ordersample1,ordersample2])
    return com1

def entropy(hsi_img):
    band_entropies = []
    h, w, c = hsi_img.shape
    for band in range(16):
        current_band = hsi_img[:, :, band]
        pixel_counts = np.histogram(current_band, bins=np.arange(256))[0]
        pixel_probabilities = pixel_counts / np.sum(pixel_counts)
        entropy = -np.sum(pixel_probabilities * np.log2(pixel_probabilities + 1e-10))
        band_entropies.append(entropy)
    return band_entropies


def normalize(data):
    min_data = min(data)
    max_data = max(data)
    normalize_data = [(x - min_data) / (max_data - min_data) for x in data]
    return normalize_data


def distance(coor_list):
    distance_matrix = np.zeros((16, 16))
    for i in range(16):
        for j in range(16):
            x1, y1 = coor_list[i]
            x2, y2 = coor_list[j]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distance_matrix[i, j] = distance
    return distance_matrix


def distance_y(coor_list, a):
    distance_matrix = np.zeros((16, 16))
    for i in range(16):
        for j in range(16):
            x1, y1 = coor_list[i]
            x2, y2 = coor_list[j]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if y1 > y2 and i not in a:
                distance = 0
            distance_matrix[i, j] = distance
    return distance_matrix


def y_coor(coor, a):
    result_list = []
    for i, row in enumerate(coor):
        if i in a:
            max_value = max(row)
            result_list.append(max_value)
        else:

            non_zero_min = min(value for value in row if value != 0)
            result_list.append(non_zero_min)
    return result_list

 def bandselect(hsi_img, i):
     hsi_img = np.array(hsi_img)
     global order,order2
     hsi_img = X2Cube(hsi_img)
     sample = hsi_img.transpose(2, 0, 1)
     i=0
     if i == 0:
         hsi_entropy = entropy(sample)
         hsi_entropy = normalize(hsi_entropy)
         coordinates = [(index, value) for index, value in enumerate(hsi_entropy)]
         distance_matrix = distance(coordinates)
         max_value = np.max(distance_matrix)
         count_list = []
         for row in distance_matrix:
             count = np.sum(row < max_value / 3)
             count = count - 1
             count_list.append(count)
         max_count = max(count_list)
         max_count_list = [index for index, value in enumerate(count_list) if value == max_count]
         count_coordinates = [(index, value) for index, value in enumerate(count_list)]
         cont_distancematrix = distance_y(count_coordinates, max_count_list)
          y = y_coor(cont_distancematrix, max_count_list)
         order = sorted(range(len(y)), key=lambda i: y[i], reverse=True)
         #order2 = sorted(range(len(hsi_entropy)), key=lambda i: hsi_entropy[i], reverse=True)
     rgb_image1= HSI2RGB(sample, order)
     #rgb_image2 = HSI2RGB(sample, order2)
     rgb_image1 = rgb_image1.transpose(1, 2, 0)
     #rgb_image2 = rgb_image2.transpose(1, 2, 0)
     #img = np.concatenate((rgb_image1, rgb_image2), axis=2)
     return rgb_image1


