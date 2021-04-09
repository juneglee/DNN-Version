import os
from typing import Tuple, List, Sequence, Callable, Dict

import cv2
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

import torch
from torch import nn, Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from torchvision.models import mobilenet_v2
# from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import KeypointRCNN

import albumentations as A
from albumentations.pytorch import ToTensorV2

df = pd.read_csv('./data/train_df.csv', index_col='image')
df.head()


class KeypointDataset(Dataset):
    def __init__(
            self,
            image_dir: os.PathLike,
            label_path: os.PathLike,
            transforms: Sequence[Callable] = None
    ) -> None:
        self.image_dir = image_dir
        self.df = pd.read_csv(label_path)
        self.transforms = transforms

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index: int) -> Tuple[Tensor, Dict]:
        image_id = self.df.iloc[index, 0]
        labels = np.array([1])
        keypoints = self.df.iloc[index, 1:].values.reshape(-1, 2).astype(np.int64)

        x1, y1 = min(keypoints[:, 0]), min(keypoints[:, 1])
        x2, y2 = max(keypoints[:, 0]), max(keypoints[:, 1])
        boxes = np.array([[x1, y1, x2, y2]], dtype=np.int64)

        image = cv2.imread(os.path.join(self.image_dir, image_id), cv2.COLOR_BGR2RGB)

        targets = {
            'image': image,
            'bboxes': boxes,
            'labels': labels,
            'keypoints': keypoints
        }

        if self.transforms is not None:
            targets = self.transforms(**targets)

        image = targets['image']
        image = image / 255.0

        targets = {
            'labels': torch.as_tensor(targets['labels'], dtype=torch.int64),
            'boxes': torch.as_tensor(targets['bboxes'], dtype=torch.float32),
            'keypoints': torch.as_tensor(
                np.concatenate([targets['keypoints'], np.ones((24, 1))], axis=1)[np.newaxis], dtype=torch.float32
            )
        }

        return image, targets

transforms = A.Compose([
    ToTensorV2()
],  bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
    keypoint_params=A.KeypointParams(format='xy')
)

def collate_fn(batch: torch.Tensor) -> Tuple:
    return tuple(zip(*batch))

trainset = KeypointDataset('./data/train_imgs/', './data/train_df.csv', transforms)
train_loader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=8, collate_fn=collate_fn)

def get_model() -> nn.Module:
    backbone = resnet_fpn_backbone('resnet101', pretrained=True)
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )

    keypoint_roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=14,
        sampling_ratio=2
    )

    model = KeypointRCNN(
        backbone,
        num_classes=2,
        num_keypoints=24,
        box_roi_pool=roi_pooler,
        keypoint_roi_pool=keypoint_roi_pooler
    )
    return model

model = get_model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)
print(model)
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    print(f'| epoch: {epoch}', end='| ')
    for i, (images, targets) in enumerate(train_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        losses = model(images, targets)

        loss = sum(loss for loss in losses.values())
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f'| epoch: {epoch} | loss: {loss.item():.4f}', end=' | ')
            for k, v in losses.items():
                print(f'{k[5:]}: {v.item():.4f}', end=' | ')
            print()

# torch.save(model_ft.state_dict(),



# df_sub = pd.read_csv(f'{prefix_dir}/data/sample_submission.csv')
# df = pd.DataFrame(columns=df_sub.columns)
# df['image'] = files
# df.iloc[:, 1:] = all_predictions
# df.head()

# model=get_model()
# model.load_state_dict(torch.load(model-e3))


# model = get_model()
# model.load_state_dict(torch.load('model-e39.pth'))
# model.eval()
# preds = model(image)
#
# submission = pd.read_csv('sample_submission.csv')
# sub = pd.DataFrame(columns=submission.columns)
# sub.iloc[:,1:]=pred
# print(sub)
# sub.to_csv('baseline_submission.csv', index=False)
