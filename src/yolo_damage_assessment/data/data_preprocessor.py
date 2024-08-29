import os
import cv2
import numpy as np
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.original_images_path = Path(config['data']['original_images_path'])
        self.processed_images_path = Path(config['data']['processed_images_path'])
        self.augmented_images_path = Path(config['data']['augmented_images_path'])
        self.target_size = tuple(config['preprocessing']['target_size'])

    def preprocess_images(self):
        logger.info("Starting image preprocessing")
        os.makedirs(self.processed_images_path, exist_ok=True)
        
        for img_path in self.original_images_path.glob('*'):
            img = cv2.imread(str(img_path))
            if img is not None:
                processed_img = self._preprocess_single_image(img)
                cv2.imwrite(str(self.processed_images_path / img_path.name), processed_img)
        
        logger.info(f"Preprocessed images saved to {self.processed_images_path}")

    def _preprocess_single_image(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, self.target_size).astype(np.float32)
        normalized = cv2.normalize(resized, None, 0, 1, cv2.NORM_MINMAX)
        return cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR)

    def augment_images(self):
        logger.info("Starting image augmentation")
        os.makedirs(self.augmented_images_path, exist_ok=True)
        
        for img_path in self.original_images_path.glob('*'):
            img = cv2.imread(str(img_path))
            if img is not None:
                augmented_images = self._augment_single_image(img)
                for i, aug_img in enumerate(augmented_images):
                    cv2.imwrite(str(self.augmented_images_path / f"aug_{img_path.stem}_{i}.jpg"), aug_img)
        
        logger.info(f"Augmented images saved to {self.augmented_images_path}")

    def _augment_single_image(self, image: np.ndarray) -> List[np.ndarray]:
        augmented = []
        for _ in range(self.config['augmentation']['num_augmented_images']):
            aug_img = image.copy()
            
            if self.config['augmentation']['horizontal_flip'] and np.random.random() < 0.5:
                aug_img = cv2.flip(aug_img, 1)
            
            if self.config['augmentation']['vertical_flip'] and np.random.random() < 0.5:
                aug_img = cv2.flip(aug_img, 0)
            
            if self.config['augmentation']['rotation_range']:
                angle = np.random.uniform(-self.config['augmentation']['rotation_range'], 
                                          self.config['augmentation']['rotation_range'])
                M = cv2.getRotationMatrix2D((aug_img.shape[1] // 2, aug_img.shape[0] // 2), angle, 1.0)
                aug_img = cv2.warpAffine(aug_img, M, (aug_img.shape[1], aug_img.shape[0]))
            
            augmented.append(aug_img)
        
        return augmented