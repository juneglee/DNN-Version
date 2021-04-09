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
    cfg.MODEL.DEVICE = 'cuda'
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument("--config-file",
        # default="../configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
        default="../configs/quick_schedules/keypoint_rcnn_R_50_FPN_inference_acc_test.yaml", # server
        # default="../configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x_test.yaml", # server
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument("--input",
        # default=["../assert/image1/*.*", "../assert/image2/*.*"],
        # default=["../assert/test/test1.jpg"],
        # default=["../assert/person/*.jpg"],
        # default=["../../dataset/img/PascalVOC2012/person100/*.jpg"], #server
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument("--webcam",
        default="test.mp4", # input
        action="store_true",
        help="Take inputs from webcam.")

    parser.add_argument(
        "--output",
        # default="../assert/output",
        # default="../assert/person100output/gpu_test",
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
    # print(cfg)
    demo = VisualizationDemo(cfg)


    # total_t = 0
    # predict_count = 0
    if args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # start_time = time.time()
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            # cuda check
            # print(torch.cuda.device_count())
            # print(torch.cuda.get_device_name(0))
            # print(torch.cuda.is_available())
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            # logger.info("predict time :{:.2f}s".format(time.time() - start_time))
            # logger.info("predict_count iteration : {}번".format(predict_count))
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
