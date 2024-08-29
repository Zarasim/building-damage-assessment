# src/models/yolov8_model.py
from ultralytics import YOLO
import yaml


class YOLOv8Model:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model = YOLO(self.config['model']['name'])

    def train(self, data_path):
        self.model.train(
            data=data_path,
            epochs=self.config['training']['epochs'],
            batch=self.config['training']['batch_size'],
            imgsz=self.config['model']['input_size']
        )
