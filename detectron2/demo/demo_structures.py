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

img_dir = "./balloon/train"
json_file = os.path.join(img_dir, "via_region_data.json") # 경로를 병합하여 새 경로 생성
# balloon/train\via_region_data.json
with open(json_file) as f:
    imgs_anns = json.load(f)
    print(imgs_anns) #json 파일 open
    # '53500107_d24b11b3c2_b.jpg104355': {
    # 'fileref': '',
    # 'size': 104355,
    # 'filename': '53500107_d24b11b3c2_b.jpg',
    # 'base64_img_data': '',
    # 'file_attributes': {},
    # 'regions': {'0':
    # {'shape_attributes': {'name':
    # 'polygon',
    # 'all_points_x': [640, 654, ...  632, 640],
    # 'all_points_y': [399, 387, ...  412, 399]},
    # 'region_attributes': {}}}}}

# print(imgs_anns.values) # <built-in method values of dict object at 0x000001AF6BF834C8>
dataset_dicts = [] # 반복문을 위한 저장 공간

for idx, v in enumerate(imgs_anns.values()):
    record = {} # 저장공간을 딕셔너리
    # {'file_name': './balloon/train\\53500107_d24b11b3c2_b.jpg', 'image_id': 60, 'height': 768, 'width': 1024} 형식

    filename = os.path.join(img_dir, v["filename"]) # json 파일에서 filename에 대한 것을 추출하여 병합
    # print(filename)
    height, width = cv2.imread(filename).shape[:2] # 슬라이스를 각각 추출하여 h,w 에 저장
    # print("height", height)
    # print("width", width)

    record["file_name"] = filename
    record["image_id"] = idx
    record["height"] = height
    record["width"] = width
    # {'file_name': './balloon/train\\53500107_d24b11b3c2_b.jpg', 'image_id': 60, 'height': 768, 'width': 1024}

    annos = v["regions"]
    # print(annos) # 실제 annotation 에 대한 정보
    # {'0':
    # {'shape_attributes':
    # {'name': 'polygon',
    # 'all_points_x': [1020, 1000, 994,..., 1032, 1020],
    # 'all_points_y': [963, 899, 841, ... 1084, 1037, 989, 963]},
    # 'region_attributes': {}}}
    objs = []
    #  각각의 정보를 하나의 리스로 전달
    for _, anno in annos.items(): # items : key와 value를 쌍으로 얻는다
        assert not anno["region_attributes"]
        anno = anno["shape_attributes"]
        # print(anno)
        px = anno["all_points_x"]
        py = anno["all_points_y"]

        # print(zip(px,py)) # <zip object at 0x000001F35B080AC8>
        # zip : 동일한 개수로 이러우진 자료형을 묶어 주는 역할으 하는 함수
        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        poly = [p for x in poly for p in x]

        # poly 변환
        # poly2 = []
        # for x in poly: #
        #     for p in x:
        #         poly2.append(p)
        #
        # print(poly1==poly2)

        # BBox 에 대한 정보를 얻는 방법
        obj = {
            "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [poly],
            "category_id": 0,
        }
        objs.append(obj)
        # print(obj)
        # {'bbox': [625, 380, 724, 477],
        # 'bbox_mode': < BoxMode.XYXY_ABS: 0 >,
        # 'segmentation': [
        #     [640.5, 399.5, 654.5, 387.5, 664.5, 382.5, 678.5, 380.5, 692.5, 380.5, 708.5, 386.5, 717.5, 397.5, 722.5,
        #      410.5, 724.5, 432.5, 714.5, 452.5, 706.5, 462.5, 694.5, 470.5, 686.5, 474.5, 665.5, 477.5, 649.5, 476.5,
        #      639.5, 476.5, 634.5, 473.5, 630.5, 476.5, 627.5, 475.5, 625.5, 473.5, 625.5, 470.5, 627.5, 466.5, 629.5,
        #      466.5, 625.5, 460.5, 625.5, 441.5, 628.5, 425.5, 632.5, 412.5, 640.5, 399.5]],
        #      'category_id': 0}

    record["annotations"] = objs
    # print(record["annotations"])
    # [{'bbox': [625, 380, 724, 477],
    # 'bbox_mode': < BoxMode.XYXY_ABS: 0 >,
    # 'segmentation': [
    #     [640.5, 399.5, 654.5, 387.5, 664.5, 382.5, 678.5, 380.5, 692.5, 380.5, 708.5, 386.5, 717.5, 397.5, 722.5, 410.5,
    #      724.5, 432.5, 714.5, 452.5, 706.5, 462.5, 694.5, 470.5, 686.5, 474.5, 665.5, 477.5, 649.5, 476.5, 639.5, 476.5,
    #      634.5, 473.5, 630.5, 476.5, 627.5, 475.5, 625.5, 473.5, 625.5, 470.5, 627.5, 466.5, 629.5, 466.5, 625.5, 460.5,
    #      625.5, 441.5, 628.5, 425.5, 632.5, 412.5, 640.5, 399.5]],
    # 'category_id': 0}]
    dataset_dicts.append(record)
    # print(dataset_dicts[0])
    # {'file_name': './balloon/train\\53500107_d24b11b3c2_b.jpg',
    # 'image_id': 60,
    # 'height': 768,
    # 'width': 1024,
    # 'annotations': [{'bbox': [625, 380, 724, 477],
    # 'bbox_mode': <BoxMode.XYXY_ABS: 0>,
    # 'segmentation': [[640.5, 399.5, ... 412.5, 640.5, 399.5]],
    # 'category_id': 0}]}]
