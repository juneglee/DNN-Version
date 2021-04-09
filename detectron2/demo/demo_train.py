import numpy as np
import pandas as pd
import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data as data_utils
from torchvision import datasets, models, transforms

from sklearn.model_selection import train_test_split

# For image-keypoints data augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

prefix_dir = "."
# train_dir = f"{prefix_dir}/data/train_imgs"

model_name = "resnet"
model_ver = "101"

num_classes = 48
batch_size = 64
num_epochs = 50
num_splits = 10
num_earlystop = 10

# resize input image
input_w = 150
input_h = 150

# optimizer
learning_rate = 0.01
feature_extract = False

df = pd.read_csv(f'{prefix_dir}/data/train_df.csv')
# print(df.head()) # index + annotation

imgs = df.iloc[:, 0].to_numpy()
# print(imgs) # 파일 이름.jpg
# print(len(imgs)) # 4195
motions = df.iloc[:, 1:] # 4195 rows x 48 columns
# print(motions)
# 행과 열에 대한 라벨 후 모두 출력
# image => 라벨로 변경
columns = motions.columns.to_list()[::2]
# print(columns) # columns list # nose_x... '
# print(len(columns)) # 24
class_labels = [label.replace('_x', '').replace('_y', '') for label in columns]
# print(class_labels) # columns에서 _x와 _y제거

keypoints = []
for motion in motions.to_numpy(): # 4195
    a_keypoints = []
    # print(len(a_keypoints)) #
    for i in range(0, motion.shape[0], 2): # 0 부터 motion.shape[0] 까지 2칸씩 띄어서
        a_keypoints.append((float(motion[i]), float(motion[i+1]))) # x, y를 묶음
        # print(a_keypoints)
    keypoints.append(a_keypoints)
keypoints = np.array(keypoints)
# print(keypoints)  # keypoints를 array를 통해서 정렬
# print(keypoints.shape) # (4195, 24, 2) # 열의 갯수, 라벨 갯수, x와 y
# print(keypoints.size) # 201360
# print(keypoints.ndim) # 3
# print(keypoints.nbytes) # 1610880
# print(type(keypoints)) # <class 'numpy.ndarray'>

def train_model(model, dataloaders, criterion, optimizer, earlysto=0, num_epochs=25, is_inception=False):
    since = time.time()