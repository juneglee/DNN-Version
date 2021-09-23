# !pip install -r requirements_hrnet_detectron.txt

import sys
sys.path.append('./detectron2_0_2_1')

import glob
import torch
from detectron2.engine import DefaultPredictor
import os


import cv2
from detectron2.config import get_cfg
# from box_model.util import mock_detector_night
import numpy as np

'''
f = open('./person_bbox_detection.txt', "w")
f_none = open('./none_person_bbox_detection.txt', "w")

cfg_bbox = get_cfg()
cfg_bbox.merge_from_file( './detectron2_0_2_1/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml')
cfg_bbox.MODEL.WEIGHTS = os.path.join('../model/detectron2/detectron_pretrained_model.pth')
cfg_bbox.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
box_model = DefaultPredictor(cfg_bbox)

image_dir = '../image/train_imgs_origin/'

for i in glob.glob(image_dir+'*.jpg'):
    img_name = i.split('/')[-1].split('.jpg')[0]
    print(img_name)

    img_temp = cv2.imread(i)
    pred_boxes, thereshold_list = mock_detector_night(box_model, img_temp)

    temp_box_size = 0
    big_box_index = -1

    for box in pred_boxes:
        box_size = (int(box[2].item()) - int(box[0].item())) * (int(box[3].item()) - int(box[1].item()))
        if box_size > temp_box_size:
            big_box_index += 1
            temp_box_size = box_size

    if len(pred_boxes) > 0:
        log = img_name + " " + str(int(pred_boxes[big_box_index][0].item())) + " " + str(int(pred_boxes[big_box_index][1].item())) + " " + str(int(pred_boxes[big_box_index][2].item() - pred_boxes[big_box_index][0].item())) + " " + str(
            int(pred_boxes[big_box_index][3].item() - pred_boxes[big_box_index][1].item())) + " " + "\n"

        f.write(log)

    else:
        log = i
        f_none.write(log)

f.close()
f_none.close()
'''

import json
import csv


def make_daycon_only_train_csv(flag, path):
    # flag == 1: ../images/train_imgs_origin
    # flag == 2: ../images/train_735

    train_img_list = []
    for i in glob.glob(path + '/*.jpg'):
        train_img_list.append(i.split('/')[-1])

    d2 = open('../annotations/person_keypoints_train.json', 'r')
    feeds = dict()
    total_ann = []
    total_img = []
    total_categories = []

    num = 0
    with d2 as json_file_daycon_d2:
        json_data = json.load(json_file_daycon_d2)

        bbox = dict()
        f = open('./person_bbox_detection.txt', 'r')
        while (True):
            line = f.readline()
            if not line: break

            bbox[line.split(' ')[0]] = [int(line.split(' ')[1]), int(line.split(' ')[2]), int(line.split(' ')[3]),
                                        int(line.split(' ')[4])]

        for i in range(len(json_data['image'])):
            if flag == 1 and num == 500:
                break

            if json_data['image'][str(i)] not in train_img_list:
                continue
            else:
                images_temp = dict()
                ann_temp = dict()
                ann_temp['num_keypoints'] = 17
                ann_temp['iscrowd'] = 0

                keypoint_temp = []

                for ind, j in enumerate(json_data):
                    if ind > 0 and ind < 35:
                        keypoint_temp.append(json_data[j][str(i)])
                        if ind % 2 == 0:
                            keypoint_temp.append(2)

                ann_temp['keypoints'] = keypoint_temp
                ann_temp['image_id'] = i
                ann_temp['id'] = i

                if json_data['image'][str(i)].split('.jpg')[0] not in bbox:
                    continue

                ann_temp['bbox'] = bbox[json_data['image'][str(i)].split('.jpg')[0]]
                ann_temp['category_id'] = 1

                images_temp['file_name'] = json_data['image'][str(i)]
                images_temp['height'] = 1080
                images_temp['width'] = 1920
                images_temp['id'] = i

                total_ann.append(ann_temp)
                total_img.append(images_temp)

                num += 1

        categories_temp = dict()
        categories_temp['supercategory'] = "person"
        categories_temp['id'] = 1
        categories_temp['keypoints'] = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                                        "left_shoulder", "right_shoulder", "left_elbow",
                                        "right_elbow", "left_wrist", "right_wrist", "left_hip",
                                        "right_hip", "left_knee", "right_knee", "left_ankle",
                                        "right_ankle"]
        categories_temp['skeleton'] = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
                                       [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6],
                                       [5, 7]]
        categories_temp['name'] = "person"
        total_categories.append(categories_temp)

        print("len(total_ann): ", len(total_ann))

        feeds['annotations'] = total_ann
        feeds['images'] = total_img
        feeds['categories'] = total_categories

    print("d2 end")

    if flag == 1:
        with open('../annotations/person_keypoints_train_daycon_small.json', "w") as json_file:
            json.dump(feeds, json_file)
    elif flag == 2:
        with open('../annotations/person_keypoints_train_daycon_735.json', "w") as json_file:
            json.dump(feeds, json_file)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from detectron2.config import get_cfg
from box_model.util import mock_detector_night
from detectron2.engine import DefaultPredictor

# import lib.init_paths
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import JointsMSELoss
from lib.core.function import train
from lib.core.function import validate
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger
from lib.utils.utils import get_model_summary

from lib.dataset import coco, mpii, coco_21, coco_4
from lib.models import pose_hrnet, pose_resnet

from types import SimpleNamespace
import csv
import time

parser = argparse.ArgumentParser(description='Train keypoints network')
# general
parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')

parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')

parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')


def train_500():
    args = parser.parse_args(
        ['--cfg', '../annotations/w48_384x288_adam_lr1e-3_finetuning.yaml', '--modelDir', '', '--logDir', '',
         '--dataDir', '',
         '--prevModelDir', ''])
    update_config(cfg, args)
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval(cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=True, flag=None
    )

    cfg_bbox = get_cfg()
    cfg_bbox.merge_from_file(
        './detectron2_0_2_1/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml')
    cfg_bbox.MODEL.WEIGHTS = os.path.join('../model/detectron2/detectron_pretrained_model.pth')
    cfg_bbox.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    box_model = DefaultPredictor(cfg_bbox)

    # copy model file
    shutil.copy2(
        os.path.join('./lib/models/', cfg.MODEL.NAME + '.py'),
        final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir='../logs'),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model, dump_input))

    model = torch.nn.DataParallel(model).cuda()

    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = eval(cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, 'train', True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), fine_tune='small'
    )

    valid_dataset = eval(cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, 'val', True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), fine_tune='small'
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 0.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        name_checkpoint = time.ctime().replace(" ", "-").replace(":", "_")

        f_output = open('../output/model_500/' + name_checkpoint + '_model.csv', 'w', newline='')
        csv_writer = csv.writer(f_output)
        csv_writer.writerow(['image', 'nose_x', 'nose_y',
                             'left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y', 'left_ear_x', 'left_ear_y',
                             'right_ear_x', 'right_ear_y',
                             'left_shoulder_x', 'left_shoulder_y', 'right_shoulder_x', 'right_shoulder_y',
                             'left_elbow_x', 'left_elbow_y',
                             'right_elbow_x', 'right_elbow_y', 'left_wrist_x', 'left_wrist_y', 'right_wrist_x',
                             'right_wrist_y', 'left_hip_x', 'left_hip_y',
                             'right_hip_x', 'right_hip_y', 'left_knee_x', 'left_knee_y', 'right_knee_x', 'right_knee_y',
                             'left_ankle_x', 'left_ankle_y',
                             'right_ankle_x', 'right_ankle_y', 'neck_x', 'neck_y', 'left_palm_x', 'left_palm_y',
                             'right_palm_x', 'right_palm_y',
                             'spine2(back)_x', 'spine2(back)_y', 'spine1(waist)_x', 'spine1(waist)_y', 'left_instep_x',
                             'left_instep_y', 'right_instep_x',
                             'right_instep_y'])

        lr_scheduler.step()

        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)

        validate(
            cfg, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir, csv_writer, 1, box_model, writer_dict
        )

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            # 'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir, name_checkpoint)