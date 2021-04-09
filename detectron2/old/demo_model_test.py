# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"

# OMP: Error #15: Initializing libiomp5.dylib
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.DEVICE = 'cpu'
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument("--config-file",
        default="../configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
        # default="../configs/quick_schedules/keypoint_rcnn_R_50_FPN_inference_acc_test.yaml", # server
        # default="../configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x_test.yaml", # server
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument("--input",
        # default=["../assert/image1/*.*", "../assert/image2/*.*"],
        default=["../assert/test/test1.jpg"],
        # default=["../assert/person/*.jpg"],
        # default=["../../dataset/img/PascalVOC2012/person100/*.jpg"], #server
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )

    parser.add_argument(
        "--output",
        default="../assert/output",
        # default="../assert/person100output/101",
        # default="../../output/keypoint/img/R_50", #server
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer

    model = build_model(cfg)  # returns a torch.nn.Module
    # print(model) # evaluation

    # DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)  # load a file, usually from cfg.MODEL.WEIGHTS
    # checkpointer = DetectionCheckpointer(model, save_dir=".")
    # checkpointer.save("model_test")  # save to output/model_999.pth
    '''
    The model files can be arbitrarily manipulated using torch.{load,save} for .pth files or pickle.{dump,load} for .pkl files.
    '''
    model = torch.load('model_test.pth') #
    print(model)
