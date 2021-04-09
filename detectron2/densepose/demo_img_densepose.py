#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# roi.heads error
# densepose_predictor.ann_index_lowres.weight/bias shape error


# python apply_net.py show configs/densepose_rcnn_R_50_FPN_s1x.yaml densepose_rcnn_R_50_FPN_s1x.pkl image.jpg dp_contour,bbox --output image_densepose_contour.png

import argparse
import glob
import logging
import os
import pickle
import sys
from typing import Any, ClassVar, Dict, List
import torch
import time
import cv2
import numpy as np

#
import multiprocessing as mp


from detectron2.config import CfgNode, get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.instances import Instances
from detectron2.utils.logger import setup_logger

#
# from predictor import VisualizationDemo
from densepose.utils.logger import verbosity_to_level

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
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# opencv error => opencv-python
OPENCV_SKIP_PYTHON_LOADER='ON'
'''
import cv2
ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.26' not found (required by /usr/anaconda3/
(ai-ats) [jklee@cwdev07 DensePose]$
'''

DOC = """
    Densepose image custom Test 
"""

LOGGER_NAME = "demo_img_densepose_test"
logger = logging.getLogger(LOGGER_NAME)

_ACTION_REGISTRY: Dict[str, "Action"] = {}

class Action(object):
    pass

def register_action(cls: type):
    """
    Decorator for action classes to automate action registration
    """
    global _ACTION_REGISTRY
    _ACTION_REGISTRY[cls.COMMAND] = cls
    return cls


class InferenceAction(Action):

    @classmethod
    def execute(cls: type, args: argparse.Namespace):
        opts = []
        cfg = cls.setup_config(args.cfg, args.model, args, opts)
        logger.info(f"Loading config from {args.cfg}")
        logger.info(f"Loading model from {args.model}")
        logger.info(f"Loading data from {args.input}")
        predictor = DefaultPredictor(cfg)
        file_list = cls._get_input_file_list(args.input)
        if len(file_list) == 0:
            logger.warning(f"No input images for {args.input}")
            return
        context = cls.create_context(args, cfg)
        total_t = 0
        num_input = 0
        for file_name in file_list:
            start_time = time.time()
            img = read_image(file_name, format="BGR")  # predictor expects BGR image.
            with torch.no_grad():
                outputs = predictor(img)["instances"]
                cls.execute_on_outputs(context, {"file_name": file_name, "image": img}, outputs)
                logger.info("Output instance time in {:.2f}s".format(time.time() - start_time))
                total_t += time.time() - start_time
                num_input += 1 # alternate len(args.input)
                # logger.info("save time {:.2f}s".format(total_t))
        cls.postexecute(context)
        logger.info("total save time {:.2f}s".format(total_t))
        logger.info("average time for 1 image {:.2f}s".format(total_t / num_input))

    @classmethod
    def setup_config(
        cls: type, config_fpath: str, model_fpath: str, args: argparse.Namespace, opts: List[str]
    ):
        cfg = get_cfg()
        add_densepose_config(cfg)
        add_hrnet_config(cfg)
        cfg.merge_from_file(config_fpath)
        cfg.merge_from_list(args.opts)
        if opts:
            cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = model_fpath
        # model.device = 'cpu'
        cfg.MODEL.DEVICE = 'cuda'
        cfg.freeze()
        return cfg

    @classmethod
    def _get_input_file_list(cls: type, input_spec: str):
        if os.path.isdir(input_spec):
            file_list = [
                os.path.join(input_spec, fname)
                for fname in os.listdir(input_spec)
                if os.path.isfile(os.path.join(input_spec, fname))
            ]
        elif os.path.isfile(input_spec):
            file_list = [input_spec]
        else:
            file_list = glob.glob(input_spec)
        return file_list


@register_action
class DumpAction(InferenceAction):
    """
    Dump action that outputs results to a pickle file
    """

    COMMAND: ClassVar[str] = "dump"

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(cls.COMMAND, help="Dump model outputs to a file.")
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(DumpAction, cls).add_arguments(parser)
        parser.add_argument(
            "--output",
            metavar="<dump_file>",
            default="results.pkl",
            help="File name to save dump to",
        )

    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        image_fpath = entry["file_name"]
        logger.info(f"Processing {image_fpath}")
        result = {"file_name": image_fpath}
        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
            if outputs.has("pred_densepose"):
                result["pred_densepose"], _ = DensePoseResultExtractor()(outputs)
        context["results"].append(result)

    @classmethod
    def create_context(cls: type, args: argparse.Namespace):
        context = {"results": [], "out_fname": args.output}
        return context

    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        out_fname = context["out_fname"]
        out_dir = os.path.dirname(out_fname)
        if len(out_dir) > 0 and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(out_fname, "wb") as hFile:
            pickle.dump(context["results"], hFile)
            logger.info(f"Output saved to {out_fname}")


@register_action
class ShowAction(InferenceAction):
    COMMAND: ClassVar[str] = "show"
    VISUALIZERS: ClassVar[Dict[str, object]] = {
        "dp_contour": DensePoseResultsContourVisualizer,
        "dp_segm": DensePoseResultsFineSegmentationVisualizer,
        "dp_u": DensePoseResultsUVisualizer,
        "dp_v": DensePoseResultsVVisualizer,
        "dp_iuv_texture": DensePoseResultsVisualizerWithTexture,
        "dp_cse_texture": DensePoseOutputsTextureVisualizer,
        "dp_vertex": DensePoseOutputsVertexVisualizer,
        "bbox": ScoredBoundingBoxVisualizer,
    }

    @classmethod
    def setup_config(
        cls: type, config_fpath: str, model_fpath: str, args: argparse.Namespace, opts: List[str]
    ):
        opts.append("MODEL.ROI_HEADS.SCORE_THRESH_TEST")
        opts.append(str(args.min_score))
        if args.nms_thresh is not None:
            opts.append("MODEL.ROI_HEADS.NMS_THRESH_TEST")
            opts.append(str(args.nms_thresh))
        cfg = super(ShowAction, cls).setup_config(config_fpath, model_fpath, args, opts)
        return cfg

    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):

        visualizer = context["visualizer"]
        extractor = context["extractor"]
        image_fpath = entry["file_name"]
        logger.info(f"Processing {image_fpath}")
        image = entry["image"]  # image를 원본 컬러로 변경
        '''
        # 해당 이미지에 대한 차원이 다르기 때문에 newaxis을 통해서 축을 증가
        # image = cv2.cvtColor(entry["image"], cv2.COLOR_BGR2GRAY)
        # print(image.shape) # (938, 626) #
        # image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        # print(image.shape) # (938, 626, 3)
        '''
        '''
        # 위의 결과를 통해서 경로를 통해 받은 이미지를 픽셀로 변경하여 error 발 생
        # image = cv2.imread(entry["image"], cv2.IMREAD_COLOR)
        # cv2.imread("Image/crow.jpg", cv2.IMREAD_COLOR)
        '''
        data = extractor(outputs)
        image_vis = visualizer.visualize(image, data)
        entry_idx = context["entry_idx"] + 1
        out_fname = cls._get_out_fname(entry_idx, context["out_fname"])
        out_dir = os.path.dirname(out_fname)
        if len(out_dir) > 0 and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        start_time = time.time()
        cv2.imwrite(out_fname, image_vis)
        logger.info(f"Output saved to {out_fname}")
        context["entry_idx"] += 1

    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        pass

    @classmethod
    def _get_out_fname(cls: type, entry_idx: int, fname_base: str):
        base, ext = os.path.splitext(fname_base)
        return base + ".{0:04d}".format(entry_idx) + ext

    @classmethod
    def create_context(cls: type, args: argparse.Namespace, cfg: CfgNode) -> Dict[str, Any]:
        vis_specs = args.visualizations.split(",")
        visualizers = []
        extractors = []
        for vis_spec in vis_specs:
            texture_atlas = get_texture_atlas(args.texture_atlas)
            texture_atlases_dict = get_texture_atlases(args.texture_atlases_map)
            vis = cls.VISUALIZERS[vis_spec](
                cfg=cfg,
                texture_atlas=texture_atlas,
                texture_atlases_dict=texture_atlases_dict,
            )
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
    args.func = ShowAction.execute
    args.input = '../../assert/input/test.jpg' # 경로 또는 디렉토리
    # args.input = '../../../dataset/img/PascalVOC2012/person100'
    args.cfg='configs/densepose_rcnn_R_101_FPN_s1x.yaml'
    args.model = 'densepose_rcnn_R_101_FPN_s1x.pkl'
    args.opts = []
    args.min_score = 0.8
    args.nms_thresh=None
    args.texture_atlas=None
    args.texture_atlases_map=None
    args.verbosity=None
    args.visualizations = 'dp_contour'
    args.output = 'assert/output/r101/densepose_img_r101.png' # fileName.0001.png
    # args.output = '../../../output/densepose/img/dp_v/densepose_img.png'  # fileName.0001.png
    verbosity = args.verbosity if hasattr(args, "verbosity") else None
    # global logger
    logger = setup_logger(name=LOGGER_NAME)

    args.func(args)


