import cv2


video_input = "../assert/input/1-10_001-C01.mp4"

video = cv2.VideoCapture(video_input)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = video.get(cv2.CAP_PROP_FPS)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

print(video)
print(frames_per_second)
print(num_frames)