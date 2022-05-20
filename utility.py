

import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import cv2
from config import config
import os

class CustomDataset(Dataset):
    def __init__(self, datasetDir, aug=True):
        self.datasetDir = datasetDir
        self.img_path_labels = glob.glob(os.path.join(datasetDir, "labels", "*"))
        self.aug = aug

    def __len__(self):
        return len(self.img_path_labels)

    def __getitem__(self, idx):
        head, tail = os.path.split(self.img_path_labels[idx])
        name, ext = os.path.splitext(tail)
        img_name = name + ".png"
        img_path = os.path.join(self.datasetDir, "images", img_name)
        image = cv2.imread(img_path)
        img_width = image.shape[1]
        img_height = image.shape[0]

        # get label from file
        file = open(self.img_path_labels[idx], "r")
        labels = []
        bbs = []
        for line in file:
            # TODO you may want to remove +1
            labels.append(int(line.split(" ")[0]) + 1)
            x_center = int(float(line.split(" ")[1]) * img_width)
            y_center = int(float(line.split(" ")[2]) * img_height)
            w = int(float(line.split(" ")[3]) * img_width)
            h = int(float(line.split(" ")[4]) * img_height)
            # top corner left
            x1 = int(x_center - w/2)
            y1 = int(y_center - h/2)
            # bottom corner right
            x2 = int(x_center + w/2)
            y2 = int(y_center + h/2)
            bb = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
            bbs.append(bb)

        target = {
            "boxes": bbs,
            "labels": labels
        }

        # apply augmentation
        if self.aug:
            augpipeline = config["augmentation"]
            image = augpipeline(image=image)
            image = image["image"]

        return image, target, img_name


def prepare_image(img):
    img = img / 255
    img = np.transpose(img, [2, 0, 1])
    img = torch.tensor(img, dtype=torch.float32)
    img = [img]
    return img
