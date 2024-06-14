import torch
import cv2
import random
from YOLOv4.models import *  # set ONNX_EXPORT in models.py
from YOLOv4.utils.datasets import *
from YOLOv4.utils.utils import *

# Initialize
device = torch_utils.select_device(device='cpu')

# Initialize model
cfg = 'YOLOv4/cfg/yolov4-relu-hat.cfg'
weights = 'YOLOv4/weights/best.pt'
names = 'YOLOv4/data/hat.names'
img_size = 416
conf_thres = 0.4
iou_thres = 0.6

model = Darknet(cfg, img_size)

# Load weights
attempt_download(weights)
if weights.endswith('.pt'):  # pytorch format
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
else:  # darknet format
    load_darknet_weights(model, weights)

# Eval mode
model.to(device).eval()

# Get names and colors
names = load_classes(names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

def detect_helmet(frame):
    #img_size = 416  # 設定圖像尺寸
    # 圖像預處理
    img = cv2.resize(frame, (img_size, img_size))
    img = torch.from_numpy(img).to(device)
    img = img.permute(2, 0, 1)  # 調整通道位置，將其從 [H, W, C] 改為 [C, H, W]
    img = img.float()  # uint8 to float32
    img /= 255.0  # 將像素值範圍從 0-255 調整到 0.0-1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # 推論
    #t1 = torch_utils.time_synchronized()
    pred = model(img, augment=False)[0]
    #t2 = torch_utils.time_synchronized()

    # 應用NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, multi_label=False, classes=None, agnostic=False)

    # 處理檢測結果
    for i, det in enumerate(pred):  # 每張圖像的檢測結果
        s = ''
        s += '%gx%g ' % img.shape[2:]  # 輸出圖像尺寸
        if det is not None and len(det):
            # 將邊界框從 img_size 縮放到原始尺寸
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            # 打印結果
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # 每類檢測結果數量
                s += '%g %ss, ' % (n, names[int(c)])  # 添加到輸出字符串

            # 寫入結果
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (names[int(cls)], conf)
                plot_one_box(xyxy, frame, label=label, color=colors[int(cls)])

    #print('%sDone. (%.3fs)' % (s, t2 - t1))

    return frame
