import numpy as np
import pandas as pd
import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data as data_utils
from torchvision import datasets, models, transforms

from sklearn.model_selection import train_test_split

# For image-keypoints data augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Prefix data directory
prefix_dir = '.'
train_dir = f'{prefix_dir}/data/train_imgs'
# model 이름 및 버전
model_name = 'resnet'
model_ver = '101'
# 데이터셋 클래스 갯수
num_classes = 48
# 배치 사이즈 (가지고 있는 메모리양에 따라 달라져야 한다)
batch_size = 64
# epoch 갯수
num_epochs = 50
num_splits = 10
num_earlystop = 10

# 이미지 resize 크기
input_w = 150
input_h = 150

# Learning rate for optimizer
learning_rate = 0.01

# Flag for feature extracting. When False, we finetune the whole model,
# when True we only update the reshaped layer params
feature_extract = False

df = pd.read_csv(f'{prefix_dir}/data/train_df.csv')
df.head()

imgs = df.iloc[:, 0].to_numpy()
motions = df.iloc[:, 1:]
columns = motions.columns.to_list()[::2]
class_labels = [label.replace('_x', '').replace('_y', '') for label in columns]
keypoints = []
for motion in motions.to_numpy():
    a_keypoints = []
    for i in range(0, motion.shape[0], 2):
        a_keypoints.append((float(motion[i]), float(motion[i+1])))
    keypoints.append(a_keypoints)
keypoints = np.array(keypoints)


def train_model(model, dataloaders, criterion, optimizer, earlystop=0, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []
    val_loss_history = []
    earlystop_value = 0

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    best_loss = 999999999

    for epoch in range(num_epochs):
        epoch_since = time.time()
        if earlystop and earlystop_value >= earlystop:
            break

        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 100)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

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
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs.float(), labels.float())
                        loss2 = criterion(aux_outputs.float(), labels.float())
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs.float(), labels.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # for regression
                running_corrects += torch.sum(outputs == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            epoch_time_elapsed = time.time() - epoch_since
            print('{} ({}) Loss: {:.4f} Acc: {:.4f} Elapsed time: {:.0f}m {:.0f}s'.format(
                phase, len(dataloaders[phase].dataset), epoch_loss, epoch_acc, epoch_time_elapsed // 60,
                                                                               epoch_time_elapsed % 60))
            #             neptune.log_metric(f'{phase}_loss', epoch_loss)
            #             neptune.log_metric(f'{phase}_acc', epoch_acc)

            # deep copy the model
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    earlystop_value = 0
                else:
                    earlystop_value += 1
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
        print()

    time_elapsed = time.time() - since
    print('Training and Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation Acc: {:4f}\n'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, {'acc': val_acc_history, 'loss': val_loss_history}

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, model_ver, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific.
    model_ft = getattr(models, f'{model_name}{model_ver}')(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    return model_ft

# Initialize the model for this run
model_ft = initialize_model(model_name, model_ver, num_classes, feature_extract, use_pretrained=True)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)

# Print the model we just instantiated
print(model_ft)

# Data augmentation and normalization for training with Albumentations
A_transforms = {
    'train':
        A.Compose([
            A.Resize(input_h, input_w, always_apply=True),
            A.OneOf([A.HorizontalFlip(p=1),
                     A.RandomRotate90(p=1),
                     A.VerticalFlip(p=1)
                     ], p=0.5),
            A.OneOf([A.MotionBlur(p=1),
                     A.GaussNoise(p=1)
                     ], p=0.5),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=True,
                                            angle_in_degrees=True)),

    'val':
        A.Compose([
            A.Resize(input_h, input_w, always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'], remove_invisible=True,
                                            angle_in_degrees=True)),

    'test':
        A.Compose([
            A.Resize(input_h, input_w, always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
}


class Dataset(data_utils.Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, data_dir, imgs, keypoints, phase, class_labels=None, data_transforms=None):
        self.data_dir = data_dir
        self.imgs = imgs
        self.keypoints = keypoints
        self.phase = phase
        self.class_labels = class_labels
        self.data_transforms = data_transforms

    def __getitem__(self, idx):
        # Read an image with OpenCV
        img = cv2.imread(os.path.join(self.data_dir, self.imgs[idx]))
        keypoints = self.keypoints[idx]

        if self.data_transforms:
            augmented = self.data_transforms[self.phase](image=img, keypoints=keypoints, class_labels=self.class_labels)
            img = augmented['image']
            keypoints = augmented['keypoints']
        keypoints = np.array(keypoints).flatten()

        return img, keypoints

    def __len__(self):
        return len(self.imgs)

# Setup the loss fxn
criterion = nn.MSELoss()

since = time.time()
X_train, X_val, y_train, y_val = train_test_split(imgs, keypoints, test_size=1/num_splits, random_state=42)
train_data = Dataset(train_dir, X_train, y_train, data_transforms=A_transforms, class_labels=class_labels, phase='train')
val_data = Dataset(train_dir, X_val, y_val, data_transforms=A_transforms, class_labels=class_labels, phase='val')
train_loader = data_utils.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = data_utils.DataLoader(val_data, batch_size=batch_size, shuffle=False)
dataloaders = {'train': train_loader, 'val': val_loader}

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate)

# Train and evaluate
# model_ft, hists = train_model(
#     model_ft, dataloaders, criterion, optimizer_ft,
#     num_epochs=num_epochs, earlystop=num_earlystop,
#     is_inception=(model_name=="inception"))
# torch.save(model_ft.state_dict(),
#            f'{prefix_dir}/local/baseline_{model_name}{model_ver}.pt')
#
# time_elapsed = time.time() - since
# print('Elapsed time: {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
#
# model_ft.load_state_dict(torch.load(f'{prefix_dir}/local/baseline_{model_name}{model_ver}.pt'))

test_dir = f'{prefix_dir}/data/test_imgs'
test_imgs = os.listdir(test_dir)


class TestDataset(data_utils.Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, data_dir, imgs, phase, data_transforms=None):
        self.data_dir = data_dir
        self.imgs = imgs
        self.phase = phase
        self.data_transforms = data_transforms

    def __getitem__(self, idx):
        filename = self.imgs[idx]
        # Read an image with OpenCV
        img = cv2.imread(os.path.join(self.data_dir, self.imgs[idx]))

        if self.data_transforms:
            augmented = self.data_transforms[self.phase](image=img)
            img = augmented['image']
        return filename, img

    def __len__(self):
        return len(self.imgs)


test_data = TestDataset(test_dir, test_imgs, data_transforms=A_transforms, phase='test')
test_loader = data_utils.DataLoader(test_data, batch_size=batch_size * 4, shuffle=False)

all_predictions = []
files = []
with torch.no_grad():
    for filenames, inputs in test_loader:
        predictions = list(model_ft(inputs.to(device)).cpu().numpy())
        print(predictions)
        files.extend(filenames)
        for prediction in predictions:
            all_predictions.append(prediction)

all_predictions = np.array(all_predictions)
for i in range(all_predictions.shape[0]):
    all_predictions[i, [2*j for j in range(num_classes//2)]] /= input_w / 1920
    all_predictions[i, [2*j + 1 for j in range(num_classes//2)]] /= input_h / 1080

df_sub = pd.read_csv(f'{prefix_dir}/data/sample_submission.csv')
df = pd.DataFrame(columns=df_sub.columns)
df['image'] = files
df.iloc[:, 1:] = all_predictions
df.head()

df.to_csv(f'{prefix_dir}/submission_{model_name}{model_ver}.csv', index=False)