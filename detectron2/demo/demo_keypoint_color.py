# # install the requirements of detectron
# !pip install pyyaml==5.1 pycocotools>=2.0.1
# import torch, torchvision
# print(torch.__version__, torch.cuda.is_available())
# !gcc --version

# :
# #install detectron2
# assert torch.__version__.startswith("1.6")
# !pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
#
# import detectron2
# from detectron2.utils.logger import setup_logger
# setup_logger()
#
# # import some common libraries
# from google.colab.patches import cv2_imshow
# import matplotlib.pyplot as plt
# import os, json, cv2, random
# import numpy as np
# import csv

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import cv2
#
# im = cv2.imread("https://www.bomb01.com/upload/news/original/0ffd961a53f39e8a9ce68e73da1fbd90.jpg")
# cfg = get_cfg()   # get a fresh new config
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
# predictor = DefaultPredictor(cfg)
# outputs = predictor(im)
# predictions = outputs["instances"].to("cpu")
#
# v = Visualizer(im[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2_imshow(out.get_image()[:, :, ::-1])
#
# im = cv2.imread("https://www.bomb01.com/upload/news/original/0ffd961a53f39e8a9ce68e73da1fbd90.jpg")
# s = im.shape
# img = np.zeros(s, np.uint8)
#
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
# predictor = DefaultPredictor(cfg)
# outputs = predictor(im)
# outputs["instances"].remove("pred_boxes")
# predictions = outputs["instances"].to("cpu")
#
# v = Visualizer(img[:,:,::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_instance_predictions(predictions)
# cv2_imshow(im)
# cv2_imshow(out.get_image()[:, :, ::-1])
#
#
# im = cv2.imread("https://www.bomb01.com/upload/news/original/0ffd961a53f39e8a9ce68e73da1fbd90.jpg")
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
# predictor = DefaultPredictor(cfg)
# outputs = predictor(im)
# outputs["instances"].remove("pred_boxes")
# predictions = outputs["instances"].to("cpu")
# keypoints = predictions.pred_keypoints
# cv2_imshow(im)
#
# x_point = []
# y_point = []
# for i in keypoints[0]:
#   x_point.append(float(i[0]))
#   y_point.append(float(i[1]))
# x = [[x_point[0],x_point[1]], [x_point[0],x_point[2]], [x_point[1],x_point[3]], [x_point[2],x_point[4]], [x_point[5],x_point[6]], [x_point[5],x_point[7]], [x_point[7],x_point[9]], [x_point[6],x_point[8]], [x_point[8],x_point[10]], [x_point[11],x_point[12]], [x_point[11],x_point[13]], [x_point[13],x_point[15]], [x_point[12],x_point[14]], [x_point[14],x_point[16]], [x_point[0],(x_point[5]+x_point[6])/2], [(x_point[5]+x_point[6])/2,(x_point[11]+x_point[12])/2]]
# y = [[y_point[0],y_point[1]], [y_point[0],y_point[2]], [y_point[1],y_point[3]], [y_point[2],y_point[4]], [y_point[5],y_point[6]], [y_point[5],y_point[7]], [y_point[7],y_point[9]], [y_point[6],y_point[8]], [y_point[8],y_point[10]], [y_point[11],y_point[12]], [y_point[11],y_point[13]], [y_point[13],y_point[15]], [y_point[12],y_point[14]], [y_point[14],y_point[16]], [y_point[0],(y_point[5]+y_point[6])/2], [(y_point[5]+y_point[6])/2,(y_point[11]+y_point[12])/2]]
#
# color = ["b","b","b","b","y","darkorange","papayawhip","darkorange","papayawhip","y","green","aqua","green","aqua","r","r"]
# for i in range(len(x)):
#   plt.plot(x[i], y[i], color=color[i])
#   plt.scatter(x[i], y[i], color="k")
#
# plt.gca().invert_yaxis()
# plt.axis('off')
# plt.show()