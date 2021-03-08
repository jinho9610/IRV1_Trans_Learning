'''
torch와 torchvision version info
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
'''

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import mtcnn
import cv2
import time
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import numpy as np

mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False,
               min_face_size=40)  # keep_all=False
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True,
              min_face_size=40)  # keep_all=True
resnet = InceptionResnetV1(pretrained='vggface2').eval()

print("select mode: ", end='')
mode = input()

if mode == 'train':
    dataset = datasets.ImageFolder('photos/train')  # photos folder path
elif mode == 'val':
    dataset = datasets.ImageFolder('photos/val')  # photos folder path

# accessing names of peoples from folder names
idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}


def collate_fn(x):
    return x[0]


def dataset_maker():
    loader = DataLoader(dataset, collate_fn=collate_fn)

    i = 1
    prev_name = ''
    for img, idx in loader:
        cur_name = idx_to_class[idx]

        if cur_name != prev_name:
            i = 1
        else:
            i += 1

        img_cropped_list, prob_list = mtcnn(img, return_prob=True)

        if img_cropped_list is not None:
            boxes, _, faces = mtcnn.detect(img, landmarks=True)
            box = boxes[0]
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
            img = cv2.resize(img, dsize=(720, 720))

            if mode == 'train':
                print('data/test_me/train/' +
                      cur_name + '/' + cur_name + str(i) + '.png')
                cv2.imwrite(
                    'data/test_me/train/' + cur_name + '/' + cur_name + str(i) + '.png', img)
            elif mode == 'val':
                print('data/test_me/val/' +
                      cur_name + '/' + cur_name + str(i) + '.png')
                cv2.imwrite(
                    'data/test_me/val/' + cur_name + '/' + cur_name + str(i) + '.png', img)

        prev_name = cur_name


if __name__ == "__main__":
    dataset_maker()
