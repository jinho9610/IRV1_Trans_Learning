from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets, transforms, utils, datasets, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import mtcnn
import cv2
import time
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import copy
import torch.nn.functional as F

mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False,
               min_face_size=40)  # keep_all=False
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True,
              min_face_size=40)  # keep_all=True

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

batch_size = 2

data_dir = 'data/test_me'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size=batch_size,
                                              shuffle=True)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

cn = len(class_names)


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.5)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = utils.make_grid(inputs)

IRV1 = InceptionResnetV1(
    pretrained='vggface2', classify=True, num_classes=cn)

layer_list = list(IRV1.children())[-5:]
IRV1 = nn.Sequential(*list(IRV1.children())[:-5])

for param in IRV1.parameters():
    param.requires_grad = False


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class normalize(nn.Module):
    def __init__(self):
        super(normalize, self).__init__()

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        return x


# class NEWRES(nn.Module):
#     def __init__(self):
#         super(NEWRES, self).__init__
#         self.layer0 = nn.Sequential(*list(IRV1.children())[:-5])
#         self.layer1 = nn.Sequential(
#             nn.AdaptiveAvgPool2d(output_size=1),
#             nn.Dropout(0.6, inplace=False),
#             nn.Linear(1792, 512, bias=False),
#             nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True),
#             nn.Linear(512, cn)
#         )

#     def forward(self, x):
#         out = self.layer0(x)
#         out = out.view(batch_size, -1)
#         out = self.layer1(out)
#         return out

IRV1.avgpool_1a = nn.AdaptiveAvgPool2d(output_size=1)
IRV1.dropout = nn.Dropout(0.6)
IRV1.last_linear = nn.Sequential(
    Flatten(),
    nn.Linear(in_features=1792, out_features=512, bias=False),
)
IRV1.last_bn = nn.BatchNorm1d(
    512, eps=0.001, momentum=0.1, affine=True)
IRV1.logits = nn.Linear(layer_list[4].in_features, len(class_names))
IRV1.softmax = nn.Softmax(dim=1)

for params in IRV1.parameters():
    print(params.requires_grad)

# # # for name, module in model_ft.named_children():
# # #     print(name)

# # # print(model_ft)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IRV1 = IRV1.to(device)
for name, module in IRV1.named_children():
    print(module)