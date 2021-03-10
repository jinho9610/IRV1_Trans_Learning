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

data_dir = 'data/test_me'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size=4,
                                              shuffle=True)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes


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

model_ft = InceptionResnetV1(
    pretrained='vggface2', classify=False, num_classes=len(class_names))

layer_list = list(model_ft.children())[-5:]
model_ft = nn.Sequential(*list(model_ft.children())[:-5])

for param in model_ft.parameters():
    param.requires_grad = False

for name, module in model_ft.named_children():
    print(name)


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


# class new_res(nn.modules):
#     def __init__(self):
#         model_ft = InceptionResnetV1(
#             pretrained='vggface2', classify=False, num_classes=len(class_names))
#         self.layer0 = nn.Sequential(*list(model_ft.children())[:-5])
#         self.layer1 = nn.Sequential(
#             nn.AdaptiveAvgPool2d(output_size=1),
#             nn.Dropout(0.6),
#             nn.Linear(1792, 512, bias=False),
#             nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True),
#             nn.Linear(512, num_classes),
#             nn.logits(x)
#         )


model_ft.avgpool_1a = nn.AdaptiveAvgPool2d(output_size=1)
model_ft.last_linear = nn.Sequential(
    Flatten(),
    nn.Linear(in_features=1792, out_features=512, bias=False),
    normalize()
)

# for name, module in model_ft.named_children():
#     print(name)

# print(model_ft)

model_ft.logits = nn.Linear(layer_list[4].in_features, len(class_names))
model_ft.softmax = nn.Softmax(dim=1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-3, momentum=0.9)
optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=1e-3)
# Decay LR by a factor of *gamma* every *step_size* epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# def train_model(model, criterion, optimizer, scheduler,
#                 num_epochs=5):
#     since = time.time()
#     FT_losses = []
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)
#     # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()   # Set model to evaluate mode
#             running_loss = 0.0
#             running_corrects = 0
#             # Iterate over data.
#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#                 # zero the parameter gradients
#                 optimizer.zero_grad()
#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)
#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
#                         scheduler.step()

#                 FT_losses.append(loss.item())
#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]
#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                 phase, epoch_loss, epoch_acc))
#             # deep copy the model
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))
#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model, FT_losses


# model_ft, FT_losses = train_model(
#     model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=5)

# torch.save(model_ft, 'finetuned_IRV1.pt')
# print(model_ft.last_linear)
# plt.figure(figsize=(10, 5))
# plt.title("FRT Loss During Training")
# plt.plot(FT_losses, label="FT loss")
# plt.xlabel("iterations")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

model_ft = torch.load('finetuned_IRV1.pt')
