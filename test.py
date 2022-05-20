import argparse
from utility import CustomDataset, prepare_image
import torch
import cv2
import time
import random
import os

parser = argparse.ArgumentParser("Test model")
parser.add_argument("--out_folder", help="Folder where to save results", type=str)
parser.add_argument("--test_folder", help="fodler containing test images", type=str)
parser.add_argument(
    "--model_path", help="Path to model checkpoint", type=str)
args = parser.parse_args()

out_folder = "out"  # args.outfolder
test_folder = os.path.join("dataset", "test")
model_path = "savedmodel"

out_folder = args.out_folder
test_folder = args.test_folder
model_path = args.model_path


model = torch.load(model_path)
model.eval()

customDatasetTest = CustomDataset(datasetDir=test_folder, aug=False)

for (img, bb, name) in customDatasetTest:
    img_copy = img.copy()

    # convert img to tensor
    img = prepare_image(img)
    predictions = model(img)

    if False:
        seenClasses = []
        if len(predictions[0]["boxes"]) >= 1:

            bb_pred = predictions[0]["boxes"][0].detach().numpy().astype("int")
            cv2.rectangle(img_copy, (bb_pred[0], bb_pred[1]),
                          (bb_pred[2], bb_pred[3]), (0, 0, 255), 2)
    else:
        seenClasses = []
        for idx, box in enumerate(predictions[0]["boxes"]):

            bb_pred = box.detach().numpy().astype("int")
            cls_pred = predictions[0]["labels"][idx].detach(
            ).numpy().astype("int")
            if cls_pred not in seenClasses:
                seenClasses.append(cls_pred)
                color = (random.randint(0, 255), random.randint(
                    0, 255), random.randint(0, 255))
                cv2.rectangle(img_copy, (bb_pred[0], bb_pred[1]),
                              (bb_pred[2], bb_pred[3]), color, 2)
                cv2.putText(img_copy, str(
                    cls_pred), (bb_pred[0], bb_pred[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1)
    name_base, ext = os.path.splitext(name)                
    out_path = os.path.join(out_folder, name_base + "_predict.png")
    cv2.imwrite(out_path, img_copy)
