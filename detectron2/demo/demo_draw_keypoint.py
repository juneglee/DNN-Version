import os
from typing import Tuple, List, Sequence, Callable, Dict
#  각 자원들의 유형에 따라 정의를 내린다.

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('./data/train_df.csv', index_col='image')
# print(df.head()) # index, 각 포인트의 갯수

def draw_keypoints(
    image: np.ndarray, # 배열 
    keypoints: np.ndarray, # 배열
    edges: List[Tuple[int, int]] = None, # 리스트
    keypoint_names: Dict[int, str] = None, # 딕셔너리
    boxes: bool = True, # 불린
    dpi: int = 200 #정수
) -> None:
    """
    Args:
        image (ndarray): [H, W, C] # Heigth, Width, Channel
        keypoints (ndarray): [N, 3] # num_point, 3(RGB)
        edges (List(Tuple(int, int))): #튜플 형태의 리스트
    """
    np.random.seed(0) # 난수를 생성
    # print("seed : ",np.random.seed(42)) # seed :  None
    colors = {k: tuple(map(int, np.random.randint(0, 255, 3))) for k in range(24)}
    # 난수를 통해 얻은 값을 반복문을 통해서 튜플 형태로 담는다
    '''
    print(colors)
    {0: (102, 179, 92), 1: (14, 106, 71), 2: (188, 20, 102), 3: (121, 210, 214), 4: (74, 202, 87), 5: (116, 99, 103),
     6: (151, 130, 149), 7: (52, 1, 87), 8: (235, 157, 37), 9: (129, 191, 187), 10: (20, 160, 203), 11: (57, 21, 252),
     12: (235, 88, 48), 13: (218, 58, 254), 14: (169, 219, 187), 15: (207, 14, 189), 16: (189, 174, 189),
     17: (50, 107, 54), 18: (243, 63, 248), 19: (130, 228, 50), 20: (134, 20, 72), 21: (166, 17, 131), 22: (88, 59, 13),
     23: (241, 249, 8)}
    '''

    # BBOX
    if boxes:
        # print('keypoint',  keypoints)
        # print("min(keypoints[:, 1]) : ", min(keypoints[:, 0]))
        # (keypoints[:, 0]) :  [1046 1041 1059 1020 1048  992 1054  956 1134 1003 1078  999 1046  995
        #  1054  983 1042 1019 1013 1067 1019 1026  998 1063]
        # (keypoints[:, 1]) :  [344 329 334 338 343 394 400 368 371 327 341 570 573 695 698 820 829 373
        #  316 335 455 514 826 838]
        x1, y1 = min(keypoints[:, 0]), min(keypoints[:, 1]) # 956, 316
        # print( x1, y1)
        x2, y2 = max(keypoints[:, 0]), max(keypoints[:, 1]) # 1134, 838
        # print(image)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 100, 91), thickness=3)
        # rectangle(이미지, 시작 좌표, 종료 좌표, 컬러, 두께)


    #
    for i, keypoint in enumerate(keypoints):
        # circle(이미지, 원의 중심 좌표, 반지름, 컬러, 두께)
        cv2.circle(
            image,
            tuple(keypoint),
            3, colors.get(i), thickness=3, lineType=cv2.FILLED)

        if keypoint_names is not None:
            # 이미지에 text 추가
            # putText(이미지, 표시할 문자열, 문자열 위치, 폰트, 폰트 사이즈, 컬러)
            cv2.putText(
                image,
                f'{i}: {keypoint_names[i]}',
                tuple(keypoint),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if edges is not None:
        for i, edge in enumerate(edges):
            # 이미지에 선을 그리기
            # Start와 End 점을 연결하여 직선을 그립니다.
            # line(이미지, 시작 좌표, 종료좌표, 컬러, 선의 두께 )
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

if __name__ == '__main__':
    keypoints = df.loc['001-1-1-01-Z17_A-0000001.jpg'].values.reshape(-1, 2)
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
    image = cv2.imread('./data/train_imgs/001-1-1-01-Z17_A-0000001.jpg', cv2.COLOR_BGR2RGB) # 배열 형태로 저장
    pix = np.array(image)
    print(image.size) # 6220800
    print(pix[300][500]) # [111 124 126] # 해당 값은 픽셀의 RGB 값을 말한다.
    # 이미지를 다루기
    # 각종 명령어는 다음과 같습니다.
    #
    # im.save('my_img.jpg')    //이미지 저장
    # im.crop((10, 20, 30, 40))    //이미지 잘라내기 (좌, 상, 우, 하 순서)
    # im.resize((512, 512))    //이미지 크기 변경하기
    # im.rotate(25)    //이미지 25도 회전하기 (60분법 사용)

    # 위에서부터 순서대로 저장, 잘라내기, 크기 변경, 회전 코드입니다.
    #
    # 크기를 변경할 때 가로와 세로 비율이 일정해야 한다면 당연히 비율로 지정하면 될 것입니다.
    #
    # width, height = im.size
    # ratio = height / width
    #
    # re_width = 512
    # im.resize((re_width, round(ratio*re_width))

    draw_keypoints(image, keypoints, edges, keypoint_names, boxes=True, dpi=400)