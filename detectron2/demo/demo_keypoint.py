import os
from typing import Tuple, List, Sequence, Callable, Dict

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn, Tensor
from torch.utils import data as data_utils
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import KeypointRCNN

import albumentations as A
from albumentations.pytorch import ToTensorV2

df = pd.read_csv('./data/train_df.csv', index_col='image')
# print(df.head())

def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    edges: List[Tuple[int, int]] = None,
    keypoint_names: Dict[int, str] = None,
    boxes: bool = True,
    dpi: int = 200
) -> None:
    """
    Args:
        image (ndarray): [H, W, C]
        keypoints (ndarray): [N, 3]
        edges (List(Tuple(int, int))):
    """
    np.random.seed(42)
    colors = {k: tuple(map(int, np.random.randint(0, 255, 3))) for k in range(24)}

    if boxes:
        x1, y1 = min(keypoints[:, 0]), min(keypoints[:, 1])
        x2, y2 = max(keypoints[:, 0]), max(keypoints[:, 1])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 100, 91), thickness=3)

    for i, keypoint in enumerate(keypoints):
        cv2.circle(
            image,
            tuple(keypoint),
            3, colors.get(i), thickness=3, lineType=cv2.FILLED)

        if keypoint_names is not None:
            cv2.putText(
                image,
                f'{i}: {keypoint_names[i]}',
                tuple(keypoint),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if edges is not None:
        for i, edge in enumerate(edges):
            cv2.line(
                image,
                tuple(keypoints[edge[0]]),
                tuple(keypoints[edge[1]]),
                colors.get(edge[0]), 3, lineType=cv2.LINE_AA)

    fig, ax = plt.subplots(dpi=dpi)
    ax.imshow(image)
    ax.axis('off')
    plt.show()
    # fig.savefig('example.png')

keypoints = df.loc['002-1-1-01-Z17_C-0000011.jpg'].values.reshape(-1, 2)
keypoints = keypoints.astype(np.int64)
keypoint_names = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle',
    17: 'neck',
    18: 'left_palm',
    19: 'right_palm',
    20: 'spine2(back)',
    21: 'spine1(waist)',
    22: 'left_instep',
    23: 'right_instep'
}

edges = [
    (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (9, 18),
    (10, 19), (5, 7), (7, 9), (11, 13), (13, 15), (12, 14),
    (14, 16), (15, 22), (16, 23), (20, 21), (5, 6), (5, 11),
    (6, 12), (11, 12), (17, 20), (20, 21),
]

image = cv2.imread('./data/train_imgs/002-1-1-01-Z17_C-0000011.jpg', cv2.COLOR_BGR2RGB)
# draw_keypoints(image, keypoints, edges, keypoint_names, boxes=False, dpi=400)

image = cv2.imread('./data/train_imgs/001-1-1-01-Z17_A-0000001.jpg', cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (1333, 800))
image = image / 255.0
image = image.transpose(2, 0, 1)
image = [torch.as_tensor(image, dtype=torch.float32)]

model = keypointrcnn_resnet50_fpn(pretrained=True, progress=False)
model.eval()
preds = model(image)
preds[0].keys()

keypoints = preds[0]['keypoints'].detach().numpy().copy()[0]
image = cv2.imread('./data/train_imgs/001-1-1-01-Z17_A-0000001.jpg', cv2.COLOR_BGR2RGB)
keypoints[:, 0] *= image.shape[1]/1333
keypoints[:, 1] *= image.shape[0]/800
keypoints = keypoints[:, :2]

edges = [
    (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
    (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
    (12, 14), (14, 16), (5, 6)
]

# draw_keypoints(image, keypoints, edges, boxes=False)

# 모델 학습 
# SOTA 모형들도 비슷하짐나 BBox 좌표가 들어갑니다.
# 박스 좌표가 없기 때문에 박스에 대한 정보를 키포인트에서 뽑음
# 나머지는 MS COCO 포맷을 사용

#
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
# 트레이닝 튜닝
train_loader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=1, collate_fn=collate_fn)
# train_loader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=8, collate_fn=collate_fn)

def get_model() -> nn.Module:
    backbone = resnet_fpn_backbone('resnet18', pretrained=True)
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


# def train(device='cuda:0'):
#     model = get_model()
#     model.to(device)
#     optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
#     num_epochs = 10
#     for epoch in range(num_epochs):
#         model.train()
#         for i, (images, targets) in enumerate(train_loader):
#             images = list(image.to(device) for image in images)
#             targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#             optimizer.zero_grad()
#             losses = model(images, targets)
#
#             loss = sum(loss for loss in losses.values())
#             loss.backward()
#             optimizer.step()
#
#             if (i + 1) % 10 == 0:
#                 print(f'| epoch: {epoch} | loss: {loss.item():.4f}', end=' | ')
#                 for k, v in losses.items():
#                     print(f'{k[5:]}: {v.item():.4f}', end=' | ')
#                 print()
#     torch.save(model.state_dict(), 'model_test.pt')

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


if __name__ == '__main__':
    # train()
    #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # epoch, loss, classifier, box_reg, keypoint, objectness, rpn_box_reg
    model_ft = get_model()
    model_ft.load_state_dict(torch.load('model_test.pt'))
    model_ft = model.to(device)
    #
    test_dir = './data/test_imgs'
    test_imgs = os.listdir(test_dir)
    # 메서드에 지정한 디레곹리 내의 모든 파일과 디렉토르이 리스트를 리턴
    #
    testset = TestDataset(test_dir, test_imgs,data_transforms=transforms, phase='test')
    test_loader = data_utils.DataLoader(testset, batch_size= 64 * 4, shuffle=False)
    # print(test_loader) # <torch.utils.data.dataloader.DataLoader object at 0x000001397E203F98>
    # # DataLoader : 머신러닝 알고리즘ㅇ르 개발하기 위해서는 데이터 전처리에 많은 노력이 필요
    # # 파이토치는 데이터를 로드하는데 쉽고 가능하다면 더 좋은 가속성을 가진 코드를 만들기 위해 많은 도구를 제공

    all_predictions = []
    files = []
    with torch.no_grad():
        for filenames, inputs in test_loader:
            print(inputs)
            # predictions = list(model_ft(inputs.to(device)).cpu().numpy())
            # # print(predictions)
            # files.extend(filenames)
            # for prediction in predictions:
            #     all_predictions.append(prediction)


    # all_predictions = np.array(all_predictions)
    # print(all_predictions)

    # submission = pd.read_csv('./sample_submission.csv')
    # submission.iloc[:, 1:] = pred * 4  # image size를 1920x1080 -> 480x270으로 바꿔서 예측했으므로 * 4
    # # submission

    # submission.to_csv('baseline.csv', index=False)

    # 이미지 시각화
    # from eval import get_model
    # image = cv2.imread('./data/test_imgs/697-3-5-34-Z94_C-0000031.jpg', cv2.COLOR_BGR2RGB)
    # image = image / 255.0
    # image = image.transpose(2, 0, 1)
    # image = [torch.as_tensor(image, dtype=torch.float32)]
    #
    # model = get_model()
    # model.load_state_dict(torch.load('model_test.pt'))
    # model.eval()
    # preds = model(image)


    # print(preds)
    # keypoints = preds[0]['keypoints'].detach().numpy().copy()[0]
    # # print(keypoints) # [9.53820068e+02 4.59206665e+02 1.00000000e+00]
    # image = cv2.imread('./data/test_imgs/697-3-5-34-Z94_C-0000031.jpg', cv2.COLOR_BGR2RGB)
    # keypoints = keypoints[:, :2]
    #
    # edges = [
    #     (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (9, 18),
    #     (10, 19), (5, 7), (7, 9), (11, 13), (13, 15), (12, 14),
    #     (14, 16), (15, 22), (16, 23), (20, 21), (5, 6), (5, 11),
    #     (6, 12), (11, 12), (17, 20), (20, 21),
    # ]

    # draw_keypoints(image, keypoints, edges, boxes=False)


