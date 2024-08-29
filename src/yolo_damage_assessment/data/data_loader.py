import os
import random
import shutil
from pathlib import Path
from typing import Tuple, List
import yaml
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config: dict):
        self.config = config
        self.processed_images_path = Path(config['data']['processed_images_path'])
        self.augmented_images_path = Path(config['data']['augmented_images_path'])
        self.train_path = Path(config['data']['train'])
        self.val_path = Path(config['data']['val'])
        self.test_path = Path(config['data']['test'])
        self.train_split = config['data']['train_split']
        self.val_split = config['data']['val_split']
        self.test_split = config['data']['test_split']

    def load_and_split_data(self) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Load all images (processed and augmented) and split them into train, validation, and test sets.
        """
        logger.info("Loading and splitting data")
        all_images = list(self.processed_images_path.glob('*')) + list(self.augmented_images_path.glob('*'))
        random.shuffle(all_images)

        total_images = len(all_images)
        train_end = int(total_images * self.train_split)
        val_end = train_end + int(total_images * self.val_split)

        train_images = all_images[:train_end]
        val_images = all_images[train_end:val_end]
        test_images = all_images[val_end:]

        logger.info(f"Split data: {len(train_images)} train, {len(val_images)} validation, {len(test_images)} test")
        return train_images, val_images, test_images

    def prepare_yolo_dataset(self):
        """
        Prepare the dataset in the format required by YOLOv8.
        """
        logger.info("Preparing YOLOv8 dataset")
        train_images, val_images, test_images = self.load_and_split_data()

        self._prepare_split(train_images, self.train_path)
        self._prepare_split(val_images, self.val_path)
        self._prepare_split(test_images, self.test_path)

        self._create_data_yaml()

        logger.info("YOLOv8 dataset preparation completed")

    def _prepare_split(self, images: List[Path], output_path: Path):
        """
        Prepare a single split (train, val, or test) by copying images and labels.
        """
        images_path = output_path / 'images'
        labels_path = output_path / 'labels'
        images_path.mkdir(parents=True, exist_ok=True)
        labels_path.mkdir(parents=True, exist_ok=True)

        for img_path in images:
            # Copy image
            shutil.copy(img_path, images_path / img_path.name)
            
            # Copy corresponding label file if it exists
            label_path = img_path.with_suffix('.txt')
            if label_path.exists():
                shutil.copy(label_path, labels_path / label_path.name)
            else:
                logger.warning(f"Label file not found for {img_path.name}")

    def _create_data_yaml(self):
        """
        Create a data.yaml file required by YOLOv8.
        """
        data_yaml = {
            'train': str(self.train_path),
            'val': str(self.val_path),
            'test': str(self.test_path),
            'nc': self.config['data']['nc'],
            'names': self.config['data']['names']
        }

        with open('data.yaml', 'w') as f:
            yaml.dump(data_yaml, f)

        logger.info("Created data.yaml for YOLOv8")

    def get_data_yaml_path(self) -> str:
        """
        Return the path to the data.yaml file.
        """
        return 'data.yaml'