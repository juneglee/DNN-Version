# instance
# python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --input image1.jpg --opts MODEL.WEIGHTS model_final_f10217.pkl

# keypoint
# python demo.py --config-file ../configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml --input image1.jpg --opts MODEL.WEIGHTS model_final_a6e10b.pkl

# BBox no
# detcetron2/ detectron2 /util/ visualizer.py

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
        # configs에서 yaml 파일로 저장되어 있는 것을 사용할 경우 quick을 이용
        # default="../configs/quick_schedules/keypoint_rcnn_R_50_FPN_3x.yaml",
        default="../configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml",
        # default="../configs/quick_schedules/keypoint_rcnn_R_50_FPN_inference_acc_test.yaml",
        # default="../configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x_test.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument("--video-input",
        # add file name
        # default="../assert/input/volleyball.mp4",
        default="../assert/input/BTS1-seg1.mp4",
        # default="../../dataset/video/keypoint/baseball_1.mp4",
        help="Path to video file."
    )

    parser.add_argument("--webcam",
        action="store_true",
        help="Take inputs from webcam."
    )

    parser.add_argument("--input",
        # list 형태로 출력해야 한다
        # webcam과 image를 동시에 사용 불가능
        # default=["../assert/image1/*.*", "../assert/image2/*.*"],
        # default=["../assert/person/*.jpg"],
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )

    parser.add_argument(
        "--output",
        default="../assert/output",
        # default="../../output/keypoint/video",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.98,
        # default=0.5,
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
    # args default
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    # print(args.input)

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    # video input
    if args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                # openh264 error => 264 에서 MPEG 변경
                fourcc=cv2.VideoWriter_fourcc(*"MPEG"),  # 디지털 미디어 포맷을 생성, 인코딩 방식 설정
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        total_t = 0.0
        num_input = 0
        assert os.path.isfile(args.video_input)
        # print("type: ", type(demo.run_on_video(video)))
        # <generator object VisualizationDemo.run_on_video at 0x00000286CFE06C50>
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            start_time = time.time()
            # print('start_time', start_time)
            if args.output:
                output_file.write(vis_frame)
                num_input += 1
                total_t += time.time() - start_time
                # print(total_t)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()

        logger.info("total time {:.2f}s".format(total_t))
        logger.info("average time for 1 image {:.2f}s".format(total_t / len(num_input)))
