#coding:utf-8
# 替换主干网络，训练
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8-efficientViT.yaml')
    model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='datasets/TomatoData/data.yaml', epochs=250, batch=4)
