# %%
import cv2
import torch
from torchsummary import summary
from PIL import Image
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import glob
from utility import prepare_image, CustomDataset
import argparse
from config import config


# %%
parser = argparse.ArgumentParser("Train model")
parser.add_argument(
    "--model_path", help="path where to save the trained model", type=str)
parser.add_argument(
    "--train_folder", help="path to folder with training images", type=str)

args = parser.parse_args()

customDatasetTrain = CustomDataset(datasetDir=args.train_folder)
img, bb, _ = customDatasetTrain[1]


NUM_CLASSES = config["num_classes"]
LEARNING_RATE = config["learning_rate"]
EPOCH = config["epoch"]

# %%
model = retinanet_resnet50_fpn(num_classes=NUM_CLASSES)

# %%
# Train loop
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

for idx_epoch in range(EPOCH):
    for idx_sample in range(customDatasetTrain.__len__()):
        # zero out gradients
        optimizer.zero_grad()

        # load image and labels
        img, target, _ = customDatasetTrain[idx_sample]

        # set model in training mode
        model.train()

        # convert img to tensor
        img = img / 255
        img = np.transpose(img, [2, 0, 1])
        img = torch.tensor(img, dtype=torch.float32)

        # convert target to tensor (one image might have multiple detections)
        n_object = len(target["boxes"])
        bbs = []
        labels = []
        for idx_object in range(n_object):
            labels.append(torch.tensor(target["labels"][idx_object]))

            x1 = target["boxes"][idx_object]["x1"]
            y1 = target["boxes"][idx_object]["y1"]
            x2 = target["boxes"][idx_object]["x2"]
            y2 = target["boxes"][idx_object]["y2"]

            bbs.append(torch.tensor([x1, y1, x2, y2], dtype=torch.float64))

        # from list to full pytorch tensors
        bbs = torch.stack(bbs)
        labels = torch.stack(labels)
        target = {
            "labels": labels,
            "boxes": bbs
        }

        # wrap tensor inside lists (one image <==> one list element)
        img = [img]
        target = [target]

        # run inference
        train_losses = model(img, target)

        # create final loss
        loss_cls = train_losses["classification"]
        loss_bb = train_losses["bbox_regression"]
        loss_total = loss_cls + loss_bb

        print(loss_total)

        # backward loss
        loss_total.backward()
        optimizer.step()

# %%
torch.save(model, args.model_path)
