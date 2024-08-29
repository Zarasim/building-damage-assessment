import pytest
import tempfile
import shutil
from pathlib import Path
import yaml
from yolo_damage_assessment.data.data_loader import DataLoader

@pytest.fixture
def temp_dir():
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)

@pytest.fixture
def config(temp_dir):
    return {
        'data': {
            'processed_images_path': Path(temp_dir) / 'processed',
            'augmented_images_path': Path(temp_dir) / 'augmented',
            'train': Path(temp_dir) / 'train',
            'val': Path(temp_dir) / 'val',
            'test': Path(temp_dir) / 'test',
            'train_split': 0.7,
            'val_split': 0.2,
            'test_split': 0.1,
            'nc': 5,
            'names': ['class1', 'class2', 'class3', 'class4', 'class5']
        }
    }

@pytest.fixture
def data_loader(config):
    return DataLoader(config)

@pytest.fixture
def sample_images(config):
    for path in [config['data']['processed_images_path'], config['data']['augmented_images_path']]:
        path.mkdir(parents=True)
        for i in range(10):
            (path / f'image_{i}.jpg').touch()
            (path / f'image_{i}.txt').touch()  # Simulating label files
    return None  # Return None as the fixture is used for side effects

@pytest.fixture
def prepared_dataset(data_loader, sample_images):
    data_loader.prepare_yolo_dataset()
    return data_loader

def test_load_and_split_data(data_loader, sample_images):
    train, val, test = data_loader.load_and_split_data()

    assert len(train) == 28  # 70% of 40 images
    assert len(val) == 8     # 20% of 40 images
    assert len(test) == 4    # 10% of 40 images

def test_prepare_yolo_dataset(prepared_dataset, config):
    assert (config['data']['train'] / 'images').exists()
    assert (config['data']['train'] / 'labels').exists()
    assert (config['data']['val'] / 'images').exists()
    assert (config['data']['val'] / 'labels').exists()
    assert (config['data']['test'] / 'images').exists()
    assert (config['data']['test'] / 'labels').exists()

def test_create_data_yaml(prepared_dataset, config):
    data_yaml_path = prepared_dataset.get_data_yaml_path()
    assert Path(data_yaml_path).exists()
    
    with open(data_yaml_path, 'r') as f:
        data_yaml = yaml.safe_load(f)
    
    assert data_yaml['nc'] == config['data']['nc']
    assert data_yaml['names'] == config['data']['names']