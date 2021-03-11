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

batch_size = 3

# directory where the dataset exists, this has 'train', 'val' directory inside
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

cn = len(class_names)  # num of class names, this will go to the last layer

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = utils.make_grid(inputs)

# get pretrained model
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


# make classfication(?) layer
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IRV1 = IRV1.to(device)
criterion = nn.CrossEntropyLoss()  # loss function
optimizer_ft = torch.optim.Adam(IRV1.parameters(), lr=1e-3)  # optimizer
exp_lr_scheduler = lr_scheduler.StepLR(
    optimizer_ft, step_size=7, gamma=0.1)  # scheduler


def train_model(model, criterion, optimizer, scheduler,
                num_epochs=5):
    since = time.time()
    FT_losses = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
    # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                FT_losses.append(loss.item())
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, FT_losses


IRV1, FT_losses = train_model(
    IRV1, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)

torch.save(IRV1, 'finetuned_IRV1.pt')
print(IRV1.last_linear)
plt.figure(figsize=(10, 5))
plt.title("FRT Loss During Training")
plt.plot(FT_losses, label="FT loss")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# # # model_ft = torch.load('finetuned_IRV1.pt')
