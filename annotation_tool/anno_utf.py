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

# with open('test.json', 'r', encoding="utf8") as file:
#      noJson = json.dumps(file, ensure_ascii=False, sort_keys=False, separators=(',', ':')).encode('utf-8')
# json_obj = 'test.json'
# test = json.dumps(json_obj, ensure_ascii=False, sort_keys=False, separators=(',', ':')).encode('utf-8')

#-*-coding:utf-8-*-
'''

# img_dir = "./person/"
# json_file = os.path.join(img_dir, "detectron2_test.json")
# with open(json_file) as f:
#     imgs_anns = json.load(f)
#
# print(json.dumps(imgs_anns, indent="\t"))

{"images": 
[{"video_no": 240059, 
"img_no": 4417812,
"img_path": "/image/1-1/1-1_601-C02/1-1_601-C02_009.jpg", 
"width": 1920, 
"height": 1080, 
"action_category": "1"}
 
 "categories": 
 [{"supercategory": "person",
"id": 1,
"name": "person", 
"keypoints": [["right_ankle_x",..., "left_wrist_yaw"]]}]
 
 "annotations": 
 [{"img_no": 4417812, 
"person_no": 1, 
"bbox": [774.0, 489.0, 961.0, 1017.0], 
"keypoints": [9.5096435547, ... , 302.4269104004],
"num_keypoint": 16}
'''

# keypoints (14개 eyes 2개, ears 2개 없음)
# right_ankle, right_knee, right_hip, left_ankle, left_knee, left_hip 6개
# head , neck 2개
# right_shoulder, right_wrist, right_elbow, left_shoulder, left_elbow, left_wrist 6 개

img_dir = "./person/"
json_file = os.path.join(img_dir, "detectron2_test.json")
with open(json_file, 'r', encoding="utf-8") as outfile:
    imgs_anns = json.load(outfile)
    # print(imgs_anns)

# print(imgs_anns.keys())
# print(json.dumps(imgs_anns, indent="\t"))
print(imgs_anns["categories"])
# print(imgs_anns.keys())



def get_keypoint_dicts(img_dir):
    # json_file = os.path.join(img_dir, "detectron2_test.json")
    json_file = os.path.join(img_dir, "person_keypoints_train2014.json")
    # def get_person_dicts(img_dir):


    with open(json_file, 'r', encoding="utf-8") as outfile:
        imgs_anns = json.load(outfile)
        # print(imgs_anns)

    # print(imgs_anns.keys())
    # print(json.dumps(imgs_anns, indent="\t"))
    print(imgs_anns.values())
    # print(enumerate(imgs_anns.values())) # <enumerate object at 0x000001402BE08510>

    dataset_dicts = []  # f annotation
    # print(imgs_anns["images"])


    for image_dict in imgs_anns["images"]:
        # print(image_dict)
        record = {}

        filename = os.path.join(img_dir, image_dict["img_path"])
        idx = image_dict["img_no"]
        height = image_dict["height"]
        width = image_dict["width"]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        # print(record)
        # dataset_dicts.append(record)
        objs = []
        for annotations_dict in imgs_anns["annotations"]:
            # print(annotations_dict)
            bbox = annotations_dict["bbox"]
            keypoints = annotations_dict["keypoints"]
            print(keypoints)
            num_keypoint = annotations_dict["num_keypoint"]
            # poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            # poly = [p for x in poly for p in x]
            obj = {
                "bbox": [bbox],
                "bbox_mode": BoxMode.XYXY_ABS,
                # "segmentation": [poly],
                "category_id": 0,
                "keypoints": [keypoints],
                "num_keypoint": [num_keypoint],
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts
    # print(dataset_dicts)

#     index += 1
#     # print(v.keys())
#     # dict_keys(['video_no', 'img_no', 'img_path', 'width', 'height', 'action_category'])
#     # dict_keys(['supercategory', 'id', 'name', 'keypoints'])
#     # dict_keys(['img_no', 'person_no', 'bbox', 'keypoints', 'num_keypoint'])
#     record = {}

'''
    print('키는 {}, 값은 {} '.format(image_id, image_dict))
    키는 0, 값은 images 
    키는 1, 값은 categories 
    키는 2, 값은 annotations 

    키는 0, 값은 info 
    키는 1, 값은 images *동일*
    키는 2, 값은 licenses 
    키는 3, 값은 annotations *동일*
    키는 4, 값은 categories *동일*
'''
#
# for v_an in image_dict:
#     index += 1
#     # print(v_an)
#     # print(index)
#     # print("second : ", idx)
#     record = {}
#     # if not v_an["img_path"]:
#     # filename = v_an["img_path"]
#
#     # print(filename)
#     height = v_an["height"]
#     width = v_an["width"]
#     # print(height,width)
#
#     # record["file_name"] = filename  # 이미지 경로와 파일 이름
#     record["image_id"] = index  # 반복문을 통해서 얻은 idx
#     record["height"] = height  # 이미지를 읽어서 얻은 Height
#     record["width"] = width  # 이미지를 읽어서 얻은 Width
#     # print(record)
#     # annos = v["regions"]  # values를 통해서 얻은 값을 annos 로 저장
#     objs = []

# enumerate : 열거 객체를 돌려줍니다.
# 파이썬에서 현제 인덱스가 몇 번째인지 확인해야 하는 경우 쉽게 작성할 수 있도록 지원해주는 함수
# list에는 enumerate라는 키 밸류를 모두 출력해 주는 함수가 잆다

# seasons = ['Spring', 'Summer', 'Fall', 'Winter']
# list(enumerate(seasons))
# [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
# list(enumerate(seasons, start=1))
# [(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
# print(image_dict)


# box, mode, segmentation, category_id
# print(annos)
# height, width = cv2.imread(filename).shape[:2]
# cv2.imread(filename)
# cv2.show()

# AttributeError: 'NoneType' object has no attribute 'shape'
# 이미지를 읽어오지 못해서 에러 발생
# TypeError: 'NoneType' object is not iterable
#

# print(height)



# load()와 loads() 차이
# 문자열을 읽을 때는 loads()를 쓴다 (file object에 사용)
# 파일을 읽을 대는 load()를 쓴다 (function에 사용 )

# error 1
# TypeError: Object of type 'TextIOWrapper' is not JSON serializable
# dumps 함수는 일반적인 obj를 json 포맷의 String 으로 serialize한다고 설명되어 있다
# 즉 딕셔너리를 json으로 변환히기 위해서는 일반저그올 tring으로 직렬화하여 전다해아 하지만. 직렬화가 정의도지 않는 byte array로 전달하여 type error가 발생하는 것이다.
# utf8 함수를 사용해서 byte array를 string으로 변환하여 수정할 수 있다.

# error 2
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa2 in position 37: invalid start byte

# error 3
# UnicodeError: UTF-16 stream does not start with BOM
# BOM(Byte Order Mark)
# 다양한 언어를 표현할 수 있도록 해주는 유니코드 인코딩에는 여러가지 방식이 있다.
# 이렇게 비슷한 방식을 사용하는 문서를 BOM(Byte Order Mark)으러 구별이 된다. 문서 맨 앞에 눈에 보이진 않는
# 특정 바이트를 넣은 다음 이것을 해석해서 어떤 인코딩 방식이 사용되었는지 알아내는 방법이 있다.

# error4
# ValueError: binary mode doesn't take an encoding argument
# 파일 열기와 관련된 파이썬 코드에 인코딩을 안 넣었다는 의미
# 더 확실하게 설명하면, UTF-8로 저장되어 있는 텍스트 파일을 열라고 시켰는데
# 파이썬이 싫다고 에러를 밷는 것이다.