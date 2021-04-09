# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"

# OMP: Error #15: Initializing libiomp5.dylib
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file) # parser
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold # if, use ROI_HEADS
    # cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold # if, use RETINANET
    # cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold #if,  use PANOPTIC_FPN
    cfg.MODEL.DEVICE = 'cuda'
    cfg.freeze()
    return cfg
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="../configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
        # default="../configs/quick_schedules/keypoint_rcnn_R_50_FPN_inference_acc_test.yaml",
        # default="../configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",  # server
        metavar="FILE",
        help="path to config file",
    )
    # parser.add_argument("--webcam", default=True, action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--webcam",  action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input",
        # default='../assert/input/volleyball.mp4',
        help="Path to video file.")
    parser.add_argument(
        "--input",
        # default=["../assert/person/*.jpg"],
        default=["../assert/input/test.jpg"],
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='../assert/output/',
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.9,
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

    cfg = setup_cfg(args) # Model upload
    demo = VisualizationDemo(cfg)
    # run_on_image 사용
    # output : (predictions, visualized_output)
    # predictions = DefaultPredictor(cfg)
    # visualized_output

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            # 평가를 사용할 때는 PIL 형태로 사용한다.
            img = read_image(path, format="BGR")
            '''
            # print(img.shape) # (938, 626, 3)
            # print(img.shape[:2]) # (938, 626)
            픽셀 형태로 값을 받기 때문에 shape을 사용 
            (C:|Users|User\git\detectron2\detectron2\data\detection_util.py)
            python package를 통해서 PIL 형태로 경롤 통해서 받고, numpy 형태로 전환하여 값을 전달한다
            즉, 픽셀형태로 전달하여 값을 전달할 수 있도록 한다
            내부에서는 픽셀 형태로 값을 전달받아 알고리즘을 통해 전환할 수 있도록 만들어졌다 . ex ResizeTransform
            '''
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            # visualized_output 의 형태
            # print(visualized_output) # <detectron2.utils.visualizer.VisImage object at 0x000002557FF25128>
            # print(visualized_output.get_image()) # array 형태의 픽셀 형태
            # print(visualized_output.get_image().shape) # (938, 626, 3)
            # cv2.imshow('test', visualized_output.get_image()) # BGR
            # cv2.imshow('test', visualized_output.get_image()[:, :, ::-1])  # RGB
            if cv2.waitKey(0) == 27:
                break  # esc to quit
            '''
            visualized_output는 visualizer.draw_instance_predictions을 통해서 얻은 결과를 사용하며,
            VisImage class의 내부 function을 통해서 출력한다 
            '''

            # predictions의 output 형태
            '''
            DefaultPredictor 를 통해서 detectron2의 output file 행태로 출력
            '''
            # print(predictions["instances"].pred_classes)
            # tensor([0], device='cuda:0')
            # print(predictions["instances"].scores)
            # tensor([0.9997], device='cuda:0')
            # print(predictions["instances"].pred_masks)
            # print(predictions["instances"].pred_boxes)
            # Boxes(tensor([[136.8211,  69.3613, 427.8311, 899.4809]], device='cuda:0'))
            # print(predictions["instances"].pred_keypoints)
            '''
            tensor([[[3.1948e+02, 1.4870e+02, 1.0905e+00],
                     [3.3435e+02, 1.3072e+02, 2.3570e+00],
                     [3.0384e+02, 1.3541e+02, 8.7515e-01],
                     [3.6016e+02, 1.3385e+02, 7.0519e-01],
                     [2.8898e+02, 1.4323e+02, 1.0285e+00],
                     [3.9145e+02, 2.4875e+02, 5.1003e-01],
                     [2.4908e+02, 2.3390e+02, 4.6290e-01],
                     [4.0475e+02, 3.7929e+02, 6.2349e-01],
                     [1.5990e+02, 3.1128e+02, 5.7282e-01],
                     [4.1336e+02, 4.9810e+02, 6.8106e-01],
                     [1.6694e+02, 2.3156e+02, 1.2637e+00],
                     [3.6016e+02, 4.6605e+02, 1.5552e-01],
                     [2.6238e+02, 4.6527e+02, 1.5711e-01],
                     [3.5078e+02, 6.6537e+02, 5.4634e-01],
                     [2.5925e+02, 6.6459e+02, 5.1399e-01],
                     [3.4374e+02, 8.3499e+02, 4.8103e-01],
                     [2.6707e+02, 8.2796e+02, 6.1025e-01]]], device='cuda:0')
            '''
            # print(predictions["instances"].pred_keypoint_heatmaps)
            '''
            tensor([[[[-10.3986, -11.0455, -12.3392,  ..., -14.5359, -12.7900, -11.9170],
                      [-11.0933, -11.7426, -13.0412,  ..., -14.3793, -12.7853, -11.9883],
                      [-12.4826, -13.1369, -14.4454,  ..., -14.0661, -12.7759, -12.1308],
                      ...,
                      [-12.7992, -13.5513, -15.0554,  ..., -13.3049, -11.7321, -10.9457],
                      [-11.5508, -12.3948, -14.0829,  ..., -12.7317, -11.2429, -10.4985],
                      [-10.9266, -11.8166, -13.5966,  ..., -12.4451, -10.9983, -10.2749]],
            
                     [[-10.1431, -11.1890, -13.2808,  ..., -12.6850, -10.9360, -10.0615],
                      [-10.3674, -11.3704, -13.3763,  ..., -12.7718, -11.1558, -10.3479],
                      [-10.8161, -11.7332, -13.5673,  ..., -12.9453, -11.5955, -10.9206],
                      ...,
                      [-11.3122, -11.8624, -12.9627,  ..., -12.3258, -11.0326, -10.3861],
                      [-10.3357, -10.8607, -11.9108,  ..., -11.4516, -10.2528,  -9.6534],
                      [ -9.8474, -10.3599, -11.3849,  ..., -11.0145,  -9.8628,  -9.2870]],
            
                     [[-10.6517, -11.1905, -12.2680,  ..., -13.3553, -11.6189, -10.7506],
                      [-11.1847, -11.7266, -12.8104,  ..., -13.2961, -11.7654, -11.0000],
                      [-12.2506, -12.7988, -13.8952,  ..., -13.1775, -12.0584, -11.4988],
                      ...,
                      [-11.9386, -12.4339, -13.4245,  ..., -11.5886, -10.4606,  -9.8966],
                      [-10.8204, -11.3202, -12.3198,  ..., -10.7440,  -9.7416,  -9.2404],
                      [-10.2613, -10.7634, -11.7675,  ..., -10.3218,  -9.3821,  -8.9123]],
            
                     ...,
            
                     [[-13.1674, -14.0153, -15.7111,  ..., -17.6245, -15.7921, -14.8759],
                      [-14.1621, -14.9861, -16.6342,  ..., -18.2868, -16.5524, -15.6852],
                      [-16.1514, -16.9278, -18.4804,  ..., -19.6112, -18.0730, -17.3039],
                      ...,
                      [-13.6591, -14.0745, -14.9054,  ..., -16.8848, -15.4608, -14.7488],
                      [-12.0867, -12.5753, -13.5525,  ..., -15.7065, -14.1472, -13.3675],
                      [-11.3006, -11.8257, -12.8761,  ..., -15.1173, -13.4904, -12.6769]],
            
                     [[-10.4599, -11.1157, -12.4273,  ..., -16.0276, -14.0319, -13.0341],
                      [-11.5288, -12.1829, -13.4912,  ..., -16.9595, -15.0156, -14.0436],
                      [-13.6666, -14.3174, -15.6190,  ..., -18.8235, -16.9829, -16.0626],
                      ...,
                      [-12.4533, -12.7158, -13.2407,  ..., -13.4449, -12.7768, -12.4427],
                      [-11.4090, -11.8011, -12.5853,  ..., -14.0050, -13.1497, -12.7221],
                      [-10.8868, -11.3437, -12.2576,  ..., -14.2851, -13.3362, -12.8618]],
            
                     [[ -9.8888, -10.6333, -12.1224,  ..., -15.4363, -13.6450, -12.7494],
                      [-10.9646, -11.7027, -13.1790,  ..., -16.4380, -14.6950, -13.8234],
                      [-13.1163, -13.8416, -15.2922,  ..., -18.4414, -16.7948, -15.9715],
                      ...,
                      [-12.0479, -12.3302, -12.8947,  ..., -14.8048, -13.8390, -13.3561],
                      [-11.0893, -11.4673, -12.2233,  ..., -14.7048, -13.5619, -12.9905],
                      [-10.6100, -11.0359, -11.8876,  ..., -14.6548, -13.4234, -12.8077]]]],
                   device='cuda:0')
            '''


