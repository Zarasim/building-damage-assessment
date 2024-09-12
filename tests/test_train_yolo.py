import pytest
import yaml
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from yolo_damage_assessment.train_yolo import load_config, train_model


@pytest.fixture
def temp_config():
    config = {
        'model': {
            'name': 'yolov8n.pt',
            'input_size': [640, 640]
        },
        'data': {
            'yaml_path': 'path/to/data.yaml',
            'test_image': 'path/to/test_image.jpg'
        },
        'training': {
            'epochs': 1,
            'batch_size': 8,
            'run_name': 'test_run'
        }
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        yaml.dump(config, temp_file)
    yield temp_file.name
    Path(temp_file.name).unlink()


def test_load_config(temp_config):
    config = load_config(temp_config)
    assert isinstance(config, dict)
    assert 'model' in config
    assert 'data' in config
    assert 'training' in config


@patch('yolo_damage_assessment.train_yolo.YOLO')
@patch('yolo_damage_assessment.train_yolo.torch.cuda.is_available', return_value=True)
def test_train_model(mock_cuda, mock_yolo, temp_config, tmp_path):
    mock_model = MagicMock()
    mock_yolo.return_value = mock_model

    train_model(temp_config, str(tmp_path))

    mock_yolo.assert_called_once_with('yolov8n.pt')
    mock_model.train.assert_called_once()
    mock_model.val.assert_called_once()
    mock_model.predict.assert_called_once()


@patch('yolo_damage_assessment.train_yolo.YOLO')
@patch('yolo_damage_assessment.train_yolo.torch.cuda.is_available', return_value=False)
def test_train_model_cpu(mock_cuda, mock_yolo, temp_config, tmp_path):
    mock_model = MagicMock()
    mock_yolo.return_value = mock_model

    train_model(temp_config, str(tmp_path))

    mock_yolo.assert_called_once_with('yolov8n.pt')
    mock_model.train.assert_called_once()
    assert mock_model.train.call_args[1]['device'] == 'cpu'


@patch('yolo_damage_assessment.train_yolo.YOLO')
@patch('yolo_damage_assessment.train_yolo.logger')
def test_train_model_exception(mock_logger, mock_yolo, temp_config, tmp_path):
    mock_yolo.side_effect = Exception("Test exception")

    train_model(temp_config, str(tmp_path))

    mock_logger.exception.assert_called_once_with("An error occurred during training: Test exception")
