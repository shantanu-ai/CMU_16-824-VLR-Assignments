import torch
import wandb
import scipy.io

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from voc_dataset import VOCDataset

from PIL import Image

from utils import *

USE_WANDB = True

# %%

# Load the Dataset - items at a particular index can be accesed by usual indexing notation (dataset[idx])
dataset = VOCDataset(
    'trainval',
    top_n=10,
    data_dir='/ocean/projects/asc170022p/shg121/PhD/CMU-Visual-Learning-Recognition/Hw1-object-localization-main/data/VOCdevkit/VOC2007/'

)

# %%

#TODO: get the image information from index 2020
idx = 2020
input = dataset[idx]

# output image result
import matplotlib.pyplot as plt
# %matplotlib inline

plt.imshow(input['image'].permute(1,2,0))

# %%

# define original_image
original_image = tensor_to_PIL(input['image'])

width, height = original_image.size

# define regions of interest
proposals = input['rois']
proposals = np.squeeze(np.array(proposals),axis=1)

# define gt_labels
gt_labels = input['gt_classes']

# define gt_boxes
gt_boxes = input['gt_boxes']


# %%

if USE_WANDB:
    wandb.init(project="vlr-hw1", reinit=True)

# %%
class_id_to_label = dict(enumerate(dataset.CLASS_NAMES))

img = wandb.Image(original_image, boxes={
    "predictions": {
        "box_data": get_box_data(gt_labels, gt_boxes),
        "class_labels": class_id_to_label,
    },
})

wandb.log({"Grount Truth BBs": img})

# %%
img = wandb.Image(original_image, boxes={
    "predictions": {
        "box_data": get_box_data(range(len(proposals)), proposals),
        "class_labels": class_id_to_label,
    },
})

wandb.log({"ROI BBs": img})