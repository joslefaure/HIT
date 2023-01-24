# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from tqdm import tqdm
import torch
# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2

# Specify counted frames path
path = 'data/AVA/keyframes/trainval'


# Specify output json path
json_path = '/data/AVA/annotations/keypoints.json'

# Keep a dictionary of models
models = {"objects": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
          "persons_and_keypoints": "COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"}

# Specify task
task = 'persons_and_keypoints'

list_of_images = os.listdir(path)
cfg = get_cfg()

# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file(models[task]))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(models[task])

predictor = DefaultPredictor(cfg)

all_outputs = []

for image in tqdm(list_of_images):
    im = cv2.imread(path+image)
    outputs = predictor(im)
    
    if task == 'persons_and_keypoints':
        for i in range(outputs["instances"].scores.shape[0]):
            all_outputs.append({'image_id': int('{}'.format(image.split('.')[0])),
                    'category_id': int(outputs["instances"].pred_classes[i].cpu()),
                    'bbox': np.array(outputs["instances"].pred_boxes)[i].cpu().numpy().tolist(),
                    'keypoints': outputs["instances"].pred_keypoints.cpu().numpy()[i].tolist(),
                    'score': float(outputs["instances"].scores[i].cpu())})
    elif task == 'objects':
        for i in range(outputs["instances"].scores.shape[0]):
            if int(outputs["instances"].pred_classes[i].cpu()) != 0: 
                all_outputs.append({'image_id': int('{}'.format(image.split('.')[0])),
                        'category_id': int(outputs["instances"].pred_classes[i].cpu()),
                        'bbox': np.array(outputs["instances"].pred_boxes)[i].cpu().numpy().tolist(),
                        'score': float(outputs["instances"].scores[i].cpu())})
             
with open(json_path, 'w') as fp:
    json.dump(all_outputs, fp)
