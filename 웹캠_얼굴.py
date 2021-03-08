'''
torch와 torchvision 버전을 잘 맞춰줘야함
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

mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False,
               min_face_size=40)  # keep_all=False
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True,
              min_face_size=40)  # keep_all=True
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def crop_mouse(ori, box, face):
    nose_x, nose_y = int(face[2][0]), int(face[2][1])
    lm_x, lm_y = int(face[3][0]), int(face[3][1])
    rm_x, rm_y = int(face[4][0]), int(face[4][1])

    img = ori[nose_y + 6: int(box[3]), lm_x: rm_x]

    rect = [(lm_x, nose_y + 6), (rm_x, int(box[3]))]

    img = cv2.resize(img, dsize=(34, 26))

    return img, rect


def webcam_face_rec(load_data):
    # loading data.pt file
    embedding_list = load_data[0]
    name_list = load_data[1]

    cam = cv2.VideoCapture(0)

    prev = 0
    while True:
        ret, frame = cam.read()
        cur = time.time()

        if not ret:
            print("fail to grab frame, try again")
            break

        # img = Image.fromarray(frame)
        img = frame
        img_cropped_list, prob_list = mtcnn(img, return_prob=True)

        if img_cropped_list is not None:
            boxes, _, faces = mtcnn.detect(img, landmarks=True)

            for i, prob in enumerate(prob_list):
                if prob > 0.80:
                    emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()

                    dist_list = []  # list of matched distances, minimum distance is used to identify the person

                    for idx, emb_db in enumerate(embedding_list):
                        dist = torch.dist(emb, emb_db).item()
                        dist_list.append(dist)

                    min_dist = min(dist_list)  # get minumum dist value
                    min_dist_idx = dist_list.index(
                        min_dist)  # get minumum dist index
                    # get name corrosponding to minimum dist
                    name = name_list[min_dist_idx]

                    box = boxes[i]

                    original_frame = frame.copy()  # storing copy of frame before drawing on it

                    if min_dist < 0.80:
                        frame = cv2.putText(frame, name+' '+str(round(min_dist, 3)), (int(box[0]), int(
                            box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

                    frame = cv2.rectangle(frame, (int(box[0]), int(
                        box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

            for face in faces:
                for i in range(len(face)):
                    frame = cv2.circle(
                        frame, (int(face[i][0]), int(face[i][1])), 1, (0, 0, 255), -1)

        cv2.imshow("IMG", frame)

        sec = cur - prev
        prev = cur
        fps = 1 / sec
        print(fps)

        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC
            print('Esc pressed, closing...')
            break

        elif k % 256 == 32:  # space to save image
            print('Enter your name :')
            name = input()

            # create directory if not exists
            if not os.path.exists('photos/'+name):
                os.mkdir('photos/'+name)

            img_name = "photos/{}/{}.jpg".format(name, int(time.time()))
            cv2.imwrite(img_name, original_frame)
            print(" saved: {}".format(img_name))

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    load_data = torch.load('data.pt')
    webcam_face_rec(load_data)
