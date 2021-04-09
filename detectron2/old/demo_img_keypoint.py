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
        default=["../assert/person/*.jpg"],
        # default=["../../dataset/img/PascalVOC2012/person100/*.jpg"], #server
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )

    parser.add_argument(
        "--output",
        # default="../assert/output",
        default="../assert/person100output/gpu_test",
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

    total_t = 0.0
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            total_t += time.time() - start_time
            logger.info(
                "{}: {} in {:.2f}s ".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
        logger.info("total time {:.2f}s".format(total_t))
        logger.info("average time for 1 image {:.2f}s".format(total_t/len(args.input)))

'''
client R_101
[02/24 20:25:54 detectron2]: total time 743.63s
[02/24 20:25:54 detectron2]: average time for 1 image 7.44s

client R_50
[02/24 20:10:40 detectron2]: total time 571.62s
[02/24 20:10:40 detectron2]: average time for 1 image 5.72s

gpu_notebook_R_50
[02/26 17:11:12 detectron2]: total time 330.68s
[02/26 17:11:12 detectron2]: average time for 1 image 3.31s

server R_101
[02/24 19:43:39 detectron2]: total time 312.35s
[02/24 19:43:39 detectron2]: average time for 1 image 3.12s

server R_50
[02/24 19:57:29 detectron2]: total time 267.15s
[02/24 19:57:29 detectron2]: average time for 1 image 2.67s
'''