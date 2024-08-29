import pytest
import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2
from yolo_damage_assessment.data.data_preprocessor import DataPreprocessor

@pytest.fixture
def temp_dir():
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)

@pytest.fixture
def config(temp_dir):
    return {
        'data': {
            'original_images_path': Path(temp_dir) / 'original',
            'processed_images_path': Path(temp_dir) / 'processed',
            'augmented_images_path': Path(temp_dir) / 'augmented',
        },
        'preprocessing': {
            'target_size': (224, 224)
        },
        'augmentation': {
            'num_augmented_images': 2,
            'horizontal_flip': True,
            'vertical_flip': False,
            'rotation_range': 15
        }
    }

@pytest.fixture
def preprocessor(config):
    return DataPreprocessor(config)

@pytest.fixture
def sample_image(config):
    os.makedirs(config['data']['original_images_path'])
    sample_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image_path = config['data']['original_images_path'] / 'sample.jpg'
    cv2.imwrite(str(image_path), sample_image)
    return image_path

def test_preprocess_images(preprocessor, config, sample_image):
    preprocessor.preprocess_images()
    processed_images = list(config['data']['processed_images_path'].glob('*'))
    assert len(processed_images) == 1
    processed_image = cv2.imread(str(processed_images[0]))
    assert processed_image.shape[:2] == tuple(config['preprocessing']['target_size'])

def test_augment_images(preprocessor, config, sample_image):
    preprocessor.augment_images()
    augmented_images = list(config['data']['augmented_images_path'].glob('*'))
    assert len(augmented_images) == config['augmentation']['num_augmented_images']