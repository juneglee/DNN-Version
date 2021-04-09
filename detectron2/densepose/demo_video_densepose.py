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

_ACTION_REGISTRY: Dict[str, "Action"] = {}

class Action(object):
    pass

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
        context = cls.create_context(args)
        for file_name in file_list:
            video = cv2.VideoCapture(args.input)
            ret, frame = video.read()
            with torch.no_grad(): # 해당 블록을 history 트래킹 x
                outputs = predictor(frame)["instances"]
                cls.execute_on_outputs(context, {"file_name": file_name, "image": frame}, outputs)
        cls.postexecute(context)

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


# @register_action
# pass
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

# @register_action
class ShowAction(InferenceAction):

    COMMAND: ClassVar[str] = "show"
    VISUALIZERS: ClassVar[Dict[str, object]] = {
        "dp_contour": DensePoseResultsContourVisualizer,
        "dp_segm": DensePoseResultsFineSegmentationVisualizer,
        "dp_u": DensePoseResultsUVisualizer,
        "dp_v": DensePoseResultsVVisualizer,
        "bbox": ScoredBoundingBoxVisualizer,
        # ---------------------------------
        "dp_iuv_texture": DensePoseResultsVisualizerWithTexture,
        "dp_cse_texture": DensePoseOutputsTextureVisualizer,
        "dp_vertex": DensePoseOutputsVertexVisualizer,
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

    # output
    @classmethod
    def execute_on_outputs(
            cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        visualizer = context["visualizer"]
        extractor = context["extractor"]
        image_fpath = entry["file_name"]
        logger.info(f"Processing {image_fpath}") # image_fpath

        def predict(img):
            opts = []
            cfg = cls.setup_config(args.cfg, args.model, args, opts)
            predictor = DefaultPredictor(cfg)
            outputs = predictor(img)['instances']
            image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
            # tile : 배열을 반복하면서 새로운 축 추가
            # newaxis : 길이가 1인 새로운 축을 추가
            '''
            # 이미지와 동일하게 변경 (image 파일 참조)
            # image = entry["image"]  # image를 원본 컬러로 변경
            # 이미지 일때는 고정된 값을 받아도 되지만 비디오 일때는 프레임 별로 전달해야 한다. 
            # print(image.shape) # (1080, 1920, 3)
            '''
            data = extractor(outputs)
            image_vis = visualizer.visualize(image, data)
            return image_vis

        entry_idx = context["entry_idx"] + 1
        out_fname = cls._get_out_fname(entry_idx, context["out_fname"])
        out_dir = os.path.dirname(out_fname)
        if len(out_dir) > 0 and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        video = cv2.VideoCapture(image_fpath)
        cap = cv2.VideoCapture(image_fpath)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(image_fpath)

        if out_dir:
            if cap.isOpened():
                num_input = 0
                total_t = 0.0
                pbar = tqdm.tqdm(total=num_frames)
                output_file = cv2.VideoWriter(
                    filename=out_fname,
                    fourcc=cv2.VideoWriter_fourcc(*'MPEG'),
                    fps=float(frames_per_second),
                    frameSize=(width, height),
                    isColor=True,
                )
                while num_input < len(pbar):
                    ret, frame = cap.read()
                    start_time = time.time()
                    if ret:
                        pbar.update(1)
                        output_file.write(predict(frame))
                        num_input += 1
                        total_t += time.time() - start_time
                        # logger.info("total iteration : {}번 and instance time :{:.2f}s".format(
                        #     num_input, time.time() - start_time))
                    else:
                        cv2.imshow(basename, predict(frame))  # 화면에 표시
                        if cv2.waitKey(1) == 27:
                            break  # esc to quit
                pbar.close()
                logger.info("total iteration: {}번, total time :{:.2f}s ".format(
                    num_input, total_t))
                logger.info("average time for 1 image {:.2f}s".format(total_t / num_input))
            else:
                print("can't open video.")  # 캡쳐 객체 초기화 실패

            ## while => for
            # if cap.isOpened():
            #     ret, frame = cap.read()
            #     for vis_frame in tqdm.tqdm(predict(frame), total=num_frames):
            #         ret, frame = cap.read() # init
            #         if args.output:
            #             output_file.write(predict(frame))
            #         else:
            #             cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
            #             cv2.imshow(basename, predict(frame))
            #             if cv2.waitKey(1) == 27:
            #                 break  # esc to quit
            # else:
            #     print("can't open video.")  # 캡쳐 객체 초기화 실패

            cap.release()  # 캡쳐 자원 반납
            output_file.release()
            cv2.destroyAllWindows()

        logger.info(f"Output saved to {out_fname}") # assert/output/f1.0001.png
        context["entry_idx"] += 1

        '''
        -- Warning message ----- densepose_result.py ----- 
        RuntimeWarning: More than 20 figures have been opened.
        Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed
        and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
        fig = plt.figure(figsize=(width_inches, height_inches), dpi=dpi)
        '''

    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        pass

    @classmethod
    def _get_out_fname(cls: type, entry_idx: int, fname_base: str):
        base, ext = os.path.splitext(fname_base)
        return base + ".{0:04d}".format(entry_idx) + ext

    @classmethod
    def create_context(cls: type, args: argparse.Namespace) -> Dict[str, Any]:
        vis_specs = args.visualizations.split(",")
        visualizers = []
        extractors = []
        for vis_spec in vis_specs:
            texture_atlas = get_texture_atlas(args.texture_atlas)
            texture_atlases_dict = get_texture_atlases(args.texture_atlases_map)
            vis = cls.VISUALIZERS[vis_spec](
                # cfg=cfg,
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
    args.cfg = 'configs/densepose_rcnn_R_101_FPN_s1x.yaml'  # config
    args.func = ShowAction.execute  # opt show, dump
    args.input = 'assert/input/PSY17s.mp4'
    # args.input = '../../../dataset/video/keypoint/golf_swing_1.mp4'
    args.min_score = 0.8
    args.model = 'densepose_rcnn_R_101_FPN_s1x.pkl'  # pkl
    args.nms_thresh = None
    args.opts = []
    args.output = "assert/output/PSY17s.mkv"
    # args.output = "../../../output/densepose/video/golf_swing_1_v.mkv"
    args.texture_atlas = None
    args.texture_atlases_map = None
    args.verbosity = None
    args.visualizations = 'dp_contour' # dp_contour/dp_segm/dp_u/dp_v/bbox
    verbosity = args.verbosity if hasattr(args, "verbosity") else None
    # global logger
    logger = setup_logger(name=LOGGER_NAME)

    args.func(args)



