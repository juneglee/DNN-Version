from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import pandas as pd
import os, json, cv2, random
import csv
import json

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode


# print(csvFilePath)
#
# imgs_anns = pd.read_csv(csvFilePath)
# print(imgs_anns.head())
img_dir = "./data"
csvFilePath = os.path.join(img_dir, "train_df.csv")
jsonFilePath = "test.json"

data = {}
with open(csvFilePath, encoding='utf-8') as csvf:
    csvReader = csv.DictReader(csvf)

    for rows in csvReader:
        key = rows["image"]
        data[key] = rows

print(data.keys()) # image
print(data.values()) # 24 keypoint

with open("test.json", 'w', encoding= 'utf-8') as jsonf:
    jsonf.write(json.dump(data, indent = 4))

with open("test.json", 'r') as f:
    json_data  = jsonf.load(f)

# print(json.dump(json_data, indent= 4))


# def get_keypoint_dicts(img_dir):