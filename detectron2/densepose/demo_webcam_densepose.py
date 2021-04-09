#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# roi.heads error
# densepose_predictor.ann_index_lowres.weight/bias shape error
# memory error
# UserWarning: The following kwargs were not used by contour: 'texture_atlas', 'texture_atlases_dict'
import argparse
import glob
import logging
import os
import pickle
import sys
from typing import Any, ClassVar, Dict, List
import torch
import cv2
import numpy as np
import tqdm
import time

#
# import multiprocessing as mp

from detectron2.config import CfgNode, get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.instances import Instances
from detectron2.utils.logger import setup_logger

#
# from predictor import VisualizationDemo
# from densepose.utils.logger import verbosity_to_level

from densepose import add_densepose_config, add_hrnet_config
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose_outputs_vertex import (
    DensePoseOutputsTextureVisualizer,
    DensePoseOutputsVertexVisualizer,
    get_texture_atlases,
)
from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)
from densepose.vis.densepose_results_textures import (
    DensePoseResultsVisualizerWithTexture,
    get_texture_atlas,
)
from densepose.vis.extractor import CompoundExtractor, DensePoseResultExtractor, create_extractor

# OMP: Error #15: Initializing libiomp5.dylib
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

DOC = """
    Densepose Video custom Test 
"""

LOGGER_NAME = "demo_video_densepose_test"
logger = logging.getLogger(LOGGER_NAME)

opts = []
config_fpath='configs/densepose_rcnn_R_101_FPN_s1x.yaml'
model_fpath='densepose_rcnn_R_101_FPN_s1x.pkl'

# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.device_count())

def setup_config():
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(config_fpath)
    if opts:
        cfg.merge_from_list(opts)
    cfg.MODEL.WEIGHTS = model_fpath
    # model.device = 'cpu'
    cfg.MODEL.DEVICE = 'cuda'
    cfg.freeze()
    return cfg

cfg = setup_config()

class Action(object):
    pass

class InferenceAction(Action):
    @classmethod
    def execute(cls: type, args: argparse.Namespace):
        predictor = DefaultPredictor(cfg)
        file_list = [args.input]
        context = cls.create_context(args)
        for file_name in file_list:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            with torch.no_grad():
                outputs = predictor(frame)["instances"]
                cls.execute_on_outputs(context, {"file_name": file_name, "image": frame}, outputs)
        cls.postexecute(context)


# @register_action
class ShowAction(InferenceAction):

    COMMAND: ClassVar[str] = "show"
    VISUALIZERS: ClassVar[Dict[str, object]] = {
        "dp_contour": DensePoseResultsContourVisualizer,
        "dp_segm": DensePoseResultsFineSegmentationVisualizer,
        "dp_u": DensePoseResultsUVisualizer,
        "dp_v": DensePoseResultsVVisualizer,
        "bbox": ScoredBoundingBoxVisualizer,
    }

    # output
    @classmethod
    def execute_on_outputs(
            cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):

        def predict(img):  # 객체를 초기화하여 각각의 instance 마다 사용할 수 있도록 내부 함수로 사용
            predictor = DefaultPredictor(cfg)
            outputs = predictor(img)['instances']
            image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
            # image = np.array(image)
            data = extractor(outputs)
            image_vis = visualizer.visualize(image, data)
            # print('image vis:', image_vis)
            return image_vis

        visualizer = context["visualizer"]
        extractor = context["extractor"]
        # image_fpath = entry["file_name"]
        image = cv2.cvtColor(entry["image"], cv2.COLOR_BGR2GRAY)
        image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        data = extractor(outputs)
        image_vis = visualizer.visualize(image, data)
        entry_idx = context["entry_idx"] + 1

        cam = cv2.VideoCapture(0)
        predict_count = 0
        total_t = 0
        while True:
            start_time = time.time()
            try:
                ret, frame = cam.read()
                predict_count += 1
                # print(predict(frame))  # prints matrix values of each framecd
                cv2.imshow('webcam', predict(frame))
                key = cv2.waitKey(1)
                total_t += time.time() - start_time
                logger.info("predict time :{:.2f}s".format(time.time() - start_time))
                logger.info("predict_count iteration : {}번".format(predict_count))
                # capture
                if key == ord('s'):
                    cv2.imwrite(filename='saved_img.jpg', img=frame)
                    ret.release()
                    img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
                    img_new = cv2.imshow("Captured Image", img_new)
                    cv2.waitKey(1650)
                    cv2.destroyAllWindows()
                    print("Processing image...")
                    img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
                    print("Converting RGB image to grayscale...")
                    gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
                    print("Converted RGB image to grayscale...")
                    print("Resizing image to 28x28 scale...")
                    img_ = cv2.resize(gray, (28, 28))
                    print("Resized...")
                    img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=img_)
                    print("Image saved!")
                    break
                elif key == ord('q'):
                    print("Turning off camera.")
                    ret.release()
                    print("Camera off.")
                    print("Program ended.")
                    cv2.destroyAllWindows()
                    break

            except(KeyboardInterrupt):
                print("Turning off camera.")
                ret.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break

    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        pass

    @classmethod
    def _get_out_fname(cls: type, entry_idx: int, fname_base: str):
        base, ext = os.path.splitext(fname_base)
        # return base + ".{0:04d}".format(entry_idx) + ext
        return base + ext

    @classmethod
    def create_context(cls: type, args: argparse.Namespace) -> Dict[str, Any]:
        vis_specs = args.visualizations.split(",")
        visualizers = []
        extractors = []
        for vis_spec in vis_specs:
            vis = cls.VISUALIZERS[vis_spec]()
            visualizers.append(vis)
            extractor = create_extractor(vis)
            extractors.append(extractor)
        visualizer = CompoundVisualizer(visualizers)
        extractor = CompoundExtractor(extractors)
        context = {
            "extractor": extractor,
            "visualizer": visualizer,
            "out_fname": args.output,
            "entry_idx": 0,
        }
        return context

if __name__ == "__main__":


    args = argparse.Namespace()
    args.cfg = 'configs/densepose_rcnn_R_50_FPN_s1x.yaml'  # config
    args.func = ShowAction.execute  # opt show, dump
    # args.input = 'assert/input/other/golf_swing_1.mp4'
    # args.input = '../../../dataset/video/keypoint/golf_swing_1.mp4'
    args.input = None
    args.webcam = ""
    args.min_score = 0.8
    args.model = 'densepose_rcnn_R_50_FPN_s1x.pkl'  # pkl
    args.nms_thresh = None
    args.opts = []
    args.output='test.png'
    args.texture_atlas = None
    args.texture_atlases_map = None
    args.verbosity = None
    args.visualizations = 'dp_contour' # dp_contour/dp_segm/dp_u/dp_v/bbox
    verbosity = args.verbosity if hasattr(args, "verbosity") else None
    # global logger
    logger = setup_logger(name=LOGGER_NAME)

    args.func(args)



