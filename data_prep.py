import argparse
import pydicom
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

argparser = argparse.ArgumentParser()
argparser.add_argument("--train-labels", type=str, required=True)
argparser.add_argument("--train-images", type=str, required=True)
argparser.add_argument("--processed-images", type=str, required=True)
argparser.add_argument("--train-val-split", type=float, default=0.8)
args = argparser.parse_args()

labels = pd.read_csv(args.train_labels)
labels = labels.drop_duplicates("patientId")
num_pneumonia_cases = labels.Target.value_counts()[1]

balanced_labels = pd.concat(
    [labels[labels.Target == 0].iloc[0:num_pneumonia_cases, :], labels[labels.Target == 1]],
    axis=0
)
balanced_labels = balanced_labels.sample(frac=1, random_state=44)

train_images = args.train_images
processed_images = args.processed_images

def img_processing():
    if args.train_val_split < 0 or args.train_val_split > 1:
        raise ValueError("The value for training / validation split should be between 0 and 1")
    train_val_split = round(balanced_labels.shape[0] * args.train_val_split)

    for parent_dir in ["train", "val"]:
        for label in range(2):
            Path(args.processed_images + f"/{parent_dir}/{label}").mkdir(parents=True, exist_ok=True)

    for c, patient_id in enumerate(tqdm(balanced_labels.patientId)):
        dcm_path = Path(args.train_images + f"/{patient_id}")
        dcm_path = dcm_path.with_suffix(".dcm")
        dcm = pydicom.read_file(dcm_path).pixel_array / 255
        dcm_arr = cv2.resize(dcm, (224, 224)).astype(np.float16)

        label = balanced_labels.Target.iloc[c]
        
        train_or_val = "train" if c < train_val_split else "val"
        np.save(args.processed_images + f"/{train_or_val}/{str(label)}/{patient_id}", dcm_arr)

if __name__ == "__main__":
    img_processing()
