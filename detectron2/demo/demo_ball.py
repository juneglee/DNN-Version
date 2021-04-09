# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode

# json 형태
'''
"34020010494_e5cb88e1c4_k.jpg1115004": {
		"fileref": "",
		"size": 1115004,
		"filename": "34020010494_e5cb88e1c4_k.jpg",
		"base64_img_data": "",
		"file_attributes": {},
		"regions": {
			"0": {
				"shape_attributes": {
					"name": "polygon",
					"all_points_x": [
						1020,
						...
						1020
					],
					"all_points_y": [
						963,
						...
						963
					]
				},
				"region_attributes": {}
			}
		}
	},
'''
# instanace annotation
def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    # print(json_file)
    # balloon/train\via_region_data.json
    # balloon/val\via_region_data.json
    with open(json_file) as f:
        imgs_anns = json.load(f)

    # print(json.dumps(imgs_anns, indent="\t"))

    dataset_dicts = [] # f annotation
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        # print(filename)
        # balloon/train\34020010494_e5cb88e1c4_k.jpg
        # balloon/val\2917282960_06beee649a_b.jpg
        height, width = cv2.imread(filename).shape[:2]
        # 이미지를 읽어서 H, W를 구함
        # print(height)

        record["file_name"] = filename # 이미지 경로와 파일 이름
        record["image_id"] = idx # 반복문을 통해서 얻은 idx
        record["height"] = height # 이미지를 읽어서 얻은 Height
        record["width"] = width # 이미지를 읽어서 얻은 Width

        annos = v["regions"] # values를 통해서 얻은 값을 annos 로 저장
        objs = []
        # bbox : bbox 를 all_points_x, y 를 통해서 max, min 를 통해서 bounding box를 구함
        # bbox_mode : XYXY_ABS
        # segmentation : poly를 x, y를 통해서 0.5를 더해서 구함
        # category_id : 0으로 통일
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                " ": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        print(objs)
        record["annotations"] = objs
        # print(record)
        dataset_dicts.append(record)
        # print(dataset_dicts)
    return dataset_dicts


for d in ["train", "val"]:
    DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
balloon_metadata = MetadataCatalog.get("balloon_train")

dataset_dicts = get_balloon_dicts("balloon/train")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    # cv2(out.get_image()[:, :, ::-1])

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("balloon_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# # cfg.MODEL.DEVICE = "cuda"
# # cfg.freeze()
# # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
# #
# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()
#
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_balloon_dicts("balloon/val")
for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=balloon_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # print(out)
    image = cv2.cvtColor(out.get_image()[:, :, ::-1],cv2.COLOR_BGR2GRAY)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()