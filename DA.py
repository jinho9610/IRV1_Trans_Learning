from facenet_pytorch import MTCNN, InceptionResnetV1
from keras.models import load_model
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import mtcnn
import cv2
import time
import os

mtcnn = MTCNN(image_size=240, margin=0, keep_all=True,
              min_face_size=40)  # keep_all=True
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def cropped_image(img_path):
    img = cv2.imread(img_path)
    img0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_cropped_list, prob_list = mtcnn(img0, return_prob=True)
    new_img1 = cv2.imread('white.png')
    new_img2 = cv2.imread('white.png')
    new_img3 = cv2.imread('white.png')
    new_img4 = cv2.imread('white.png')

    if img_cropped_list is not None:
        boxes, _, faces = mtcnn.detect(img0, landmarks=True)

        for i, prob in enumerate(prob_list):
            box = boxes[i]
            cropped = img[int(box[1])-50: int(box[3])+50,
                          int(box[0])-30: int(box[2])+30]

            cropped = cv2.resize(cropped, dsize=(0, 0), fx=0.3, fy=0.3)
            x_offset = y_offset = 0

            new_img1[y_offset+90:y_offset+90 + cropped.shape[0],
                     x_offset:x_offset + cropped.shape[1]] = cropped             # 왼족

            new_img2[y_offset:y_offset + cropped.shape[0],
                     x_offset+150:x_offset+150 + cropped.shape[1]] = cropped  # 위쪽

            new_img3[y_offset+90:y_offset+90 + cropped.shape[0],
                     500-cropped.shape[1]:500] = cropped               # 오른쪽

            new_img4[500-cropped.shape[0]: 500,
                     x_offset+150:x_offset+150 + cropped.shape[1]] = cropped  # 아래쪽

    return new_img1, new_img2, new_img3, new_img4


def make_cropped_photos():
    dir_list = os.listdir('photos')

    for dir_name in dir_list:
        dir_path = os.path.join('photos', dir_name)
        imgs = os.listdir(dir_path)
        for img in imgs:
            if img[:2] == 'd_':
                continue
            img_path = os.path.join(dir_path, img)
            tmp1, tmp2, tmp3, tmp4 = cropped_image(img_path)
            distored_img_path1 = dir_path + '\\c1_' + img
            distored_img_path2 = dir_path + '\\c2_' + img
            distored_img_path3 = dir_path + '\\c3_' + img
            distored_img_path4 = dir_path + '\\c4_' + img
            cv2.imwrite(distored_img_path1, tmp1)
            cv2.imwrite(distored_img_path2, tmp2)
            cv2.imwrite(distored_img_path3, tmp3)
            cv2.imwrite(distored_img_path4, tmp4)


make_cropped_photos()
